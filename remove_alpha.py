from PIL import Image
import os

def pixel_alpha_composite(fg_rgba, bg_rgb):
    # Calculate the alpha value
    alpha = fg_rgba[3] / 255.0

    # Convert the RGB values
    bg_rgb = [int(round(val * (1 - alpha) + alpha * val_2)) for val, val_2 in zip(bg_rgb, fg_rgba[:3])]

    return tuple(bg_rgb)

def convert_rgba_to_rgb(foreground_path, background_path):
    # Load images
    foreground = Image.open(foreground_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")

    # Create a new image for the adjusted foreground
    adjusted_foreground = Image.new('RGB', foreground.size, (127, 127, 127))

    for x in range(foreground.width):
        for y in range(foreground.height):
            fg_rgba = foreground.getpixel((x, y))

            if fg_rgba[3] == 0:
                # Set to (127, 127, 127)
                continue
            elif fg_rgba[3] == 255:
                # Keep original RGB values
                adjusted_foreground.putpixel((x, y), fg_rgba[:3])
            else:
                # print("Shadow pixel")
                # Shadow pixel
                bg_rgb = background.getpixel((x, y))
                background.putpixel((x, y), pixel_alpha_composite(fg_rgba, bg_rgb))

    return adjusted_foreground, background


# Example usage
foreground_paths = sorted(os.listdir("inputs/foreground/with_alpha"))  # Replace with your input image path
for foreground_path in foreground_paths[3:4]:
    background_path = os.path.join("/home/michaelch/IC-Light/inputs/foreground/wooden_chair_noalpha.jpg")  # Replace with your desired output path
    foreground_path = os.path.join("inputs/foreground/with_alpha", foreground_path)

    adjusted_foreground, background = convert_rgba_to_rgb(foreground_path, background_path)

    # Save or display the result
    adjusted_foreground.show('adjusted_foreground')  # or combined_image.save('output.png')
    background.show('adjusted background')  # or combined_image.save('output.png')