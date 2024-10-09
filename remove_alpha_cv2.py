import cv2
import numpy as np
import os


def move_shadow_to_background(foreground, background):
    # Ensure the images have an alpha channel
    if foreground.shape[2] != 4:
        raise ValueError("Foreground image does not have an alpha channel")

    # Create a new image for the adjusted foreground
    adjusted_foreground = np.full(foreground.shape, (127, 127, 127, 255), dtype=np.uint8)

    # Separate the alpha channel
    alpha_channel = foreground[:, :, 3] / 255.0

    # Fully opaque pixels
    mask_opaque = alpha_channel == 1

    # Shadow pixels
    mask_shadow = (alpha_channel > 0) & (alpha_channel < 1)

    # Keep original RGB values for fully opaque pixels
    adjusted_foreground[mask_opaque] = foreground[mask_opaque]

    # Composite shadow pixels
    bg_rgb = background[:, :, :3]
    fg_rgb = foreground[:, :, :3]
    alpha_channel_expanded = alpha_channel[:, :, np.newaxis]
    composite_rgb = (1 - alpha_channel_expanded) * bg_rgb + alpha_channel_expanded * fg_rgb
    background[mask_shadow] = composite_rgb[mask_shadow]

    return adjusted_foreground[:, :, :3], background

# Example usage
foreground_paths = sorted(os.listdir("inputs/foreground/with_alpha"))  # Replace with your input image path
for foreground_path in foreground_paths[3:4]:
    background_path = os.path.join("/home/michaelch/IC-Light/inputs/foreground/wooden_chair_noalpha.jpg")  # Replace with your desired output path
    foreground_path = os.path.join("inputs/foreground/with_alpha", foreground_path)

    # Load images
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
    background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    foreground = cv2.resize(foreground, None, fx=0.5, fy=0.5)
    background = cv2.resize(background, None, fx=0.5, fy=0.5)
    cv2.imshow('Original Foreground', foreground)
    cv2.imshow('Original Background', background)

    adjusted_foreground, background = move_shadow_to_background(foreground, background)

    # Save or display the final result
    cv2.imshow('Final Adjusted Foreground', adjusted_foreground)
    cv2.imshow('Final Adjusted Background', background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()