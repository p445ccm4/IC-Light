import os
import math

import cv2
import numpy as np
import torch
import safetensors.torch as sf

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file

# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fbc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    original_height, original_width = image.shape[:2]
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    cropped_image = resized_image[top:bottom, left:right]
    return cropped_image


def resize_without_crop(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height))


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    bg_source = BGSource(bg_source)

    if bg_source == BGSource.UPLOAD:
        pass
    elif bg_source == BGSource.UPLOAD_FLIP:
        input_bg = np.fliplr(input_bg)
    elif bg_source == BGSource.GREY:
        input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(224, 32, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(32, 224, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(224, 32, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(32, 224, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong background source!'

    rng = torch.Generator(device=device).manual_seed(seed)

    # fg = reduce_and_paste(input_fg, image_width, image_height)
    # bg = resize_and_center_crop(input_bg, image_width, image_height)
    fg, bg = input_fg, input_bg
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    latents = t2i_pipe(
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg)
    results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    return results + extra_images


@torch.inference_mode()
def process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg, sigma=16)

    print('left ...')
    left = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.LEFT.value)[0][0]

    print('right ...')
    right = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.RIGHT.value)[0][0]

    print('bottom ...')
    bottom = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.BOTTOM.value)[0][0]

    print('top ...')
    top = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.TOP.value)[0][0]

    inner_results = [left * 2.0 - 1.0, right * 2.0 - 1.0, bottom * 2.0 - 1.0, top * 2.0 - 1.0]

    ambient = (left + right + bottom + top) / 4.0
    h, w, _ = ambient.shape
    matting = resize_and_center_crop((matting[..., 0] * 255.0).clip(0, 255).astype(np.uint8), w, h).astype(np.float32)[..., None] / 255.0

    def safa_divide(a, b):
        e = 1e-5
        return ((a + e) / (b + e)) - 1.0

    left = safa_divide(left, ambient)
    right = safa_divide(right, ambient)
    bottom = safa_divide(bottom, ambient)
    top = safa_divide(top, ambient)

    u = (right - left) * 0.5
    v = (top - bottom) * 0.5

    sigma = 10.0
    u = np.mean(u, axis=2)
    v = np.mean(v, axis=2)
    h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
    z = np.zeros_like(h)

    normal = np.stack([u, v, h], axis=2)
    normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
    normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)

    results = [normal, left, right, bottom, top] + inner_results
    results = [(x * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for x in results]
    return results


class BGSource(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"

def preprocess(foreground, background, target_width, target_height):
    # Reduce the size of the image by a scale factor
    original_height, original_width, original_channel = foreground.shape
    scale_factor = max(target_width / original_width, target_height / original_height) / 3
    resized_width = int(original_width * scale_factor)
    resized_height = int(original_height * scale_factor)
    resized_foreground = cv2.resize(foreground, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    resized_background = resize_and_center_crop(background, target_width, target_height)

    # Create a new image with the target size and a gray background
    new_foreground = np.full((target_height, target_width, 3), (127, 127, 127), dtype=np.uint8)

    # Calculate the position to paste the resized image in the middle-bottom
    paste_x = (target_width - resized_width) // 2
    paste_y = target_height - int(resized_height * 1.1)

    # Ensure the paste coordinates are within bounds
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)

    # Ensure the images have an alpha channel
    if original_channel == 4:
        # Separate the alpha channel
        alpha_channel = resized_foreground[:, :, 3] / 255.0

        # Fully opaque pixels
        mask_opaque = alpha_channel == 1

        # Shadow pixels
        mask_shadow = (alpha_channel > 0) & (alpha_channel < 1)

        # Keep original RGB values for fully opaque pixels
        new_foreground[paste_y:paste_y + resized_height, paste_x:paste_x + resized_width][mask_opaque] = resized_foreground[:, :, :3][mask_opaque]

        # Composite shadow pixels
        bg_rgb = resized_background[paste_y:paste_y + resized_height, paste_x:paste_x + resized_width, :3]
        fg_rgb = resized_foreground[:, :, :3]
        alpha_channel_expanded = alpha_channel[:, :, np.newaxis]
        composite_rgb = (1 - alpha_channel_expanded) * bg_rgb + alpha_channel_expanded * fg_rgb
        resized_background[paste_y:paste_y + resized_height, paste_x:paste_x + resized_width][mask_shadow] = composite_rgb[mask_shadow]

    else:
        # Paste the resized image onto the new image
        new_foreground[paste_y:paste_y + resized_height, paste_x:paste_x + resized_width] = resized_foreground

    new_foreground = cv2.cvtColor(new_foreground, cv2.COLOR_BGR2RGB)
    resized_background = cv2.cvtColor(resized_background, cv2.COLOR_BGR2RGB)
    return new_foreground, resized_background

# configs
fg_dir = "inputs/foreground/with_alpha"
bg_dir = "inputs/background"
output_dir = "outputs"
input_fg_paths = [os.path.join(fg_dir, image) for image in sorted(os.listdir(fg_dir))]
# input_fg_paths = ["/home/michaelch/IC-Light/inputs/foreground/blue_sofa_chair.png"]
input_bg_paths = [os.path.join(bg_dir, image) for image in sorted(os.listdir(bg_dir))]

for i, input_fg_path in enumerate(input_fg_paths):
    if i != 0:
        continue
    for j, input_bg_path in enumerate(input_bg_paths):
        if j != 0:
            continue
        if os.path.isdir(input_bg_path) or os.path.isdir(input_fg_path):
            continue
        prompt = "indoor, natural lighting, realistic"
        bg_source = BGSource.UPLOAD.value
        num_samples = 1
        a_prompt = 'best quality'
        n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
        seed = 12345
        steps = 20
        cfg = 7.0
        highres_scale = 1.0
        highres_denoise = 0.5

        # inference
        input_bg = cv2.imread(input_bg_path)
        input_fg = cv2.imread(input_fg_path, cv2.IMREAD_UNCHANGED)
        image_height, image_width, _ = input_bg.shape
        image_width = int(image_width // 64 * 64)
        image_height = int(image_height // 64 * 64)
        input_fg, input_bg = preprocess(input_fg, input_bg, image_width, image_height)

        ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg,
               highres_scale, highres_denoise, bg_source]

        # upload bg
        results = process_relight(*ips)
        # display and save results
        for img, mode in zip(results, ['result', 'cropped_fg', 'bg']):
            bgr = img[:, :, ::-1]
            cv2.imshow("result", bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_path = os.path.join(output_dir, f"{i}_{j}_{mode}.png")
            # cv2.imwrite(save_path, bgr)

    # # normal bg
    # results = process_normal(*ips)
    # # display and save results
    # for img, mode in zip(results[5:], ['left', 'right', 'bottom', 'top']):
    #     bgr = img[:, :, ::-1]
    #     # cv2.imshow("result", bgr)
    #     # cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     save_path = os.path.join(output_dir, f"{i}_{mode}.png")
    #     cv2.imwrite(save_path, bgr)
