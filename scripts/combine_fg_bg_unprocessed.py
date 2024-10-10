import os

import cv2
import numpy as np


def combine_fg_bg(foreground, background):
    # Create a mask for the grey background (RGB: 127, 127, 127)
    grey_background = np.array([127, 127, 127], dtype=np.uint8)
    mask = cv2.inRange(foreground, grey_background, grey_background)

    # Invert the mask to get the object (not grey)
    mask_inv = cv2.bitwise_not(mask)

    # Get the object part from the foreground
    object_part = cv2.bitwise_and(foreground, foreground, mask=mask_inv)

    # Prepare the background by resizing it to match the foreground dimensions
    background_resized = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    # Get the background part where the object will be placed
    background_part = cv2.bitwise_and(background_resized, background_resized, mask=mask)

    # Combine the object part and the background part
    combined = cv2.add(object_part, background_part)

    return combined

# Load images
root_dir = "../outputs/09102024"
output_dir = "../outputs/09102024/combined"
img_paths = os.listdir(root_dir)
fg_paths = sorted(path for path in img_paths if path.endswith("_cropped_fg.png"))
bg_paths = sorted(path for path in img_paths if path.endswith("_bg.png"))

for fg, bg in zip(fg_paths, bg_paths):
    foreground = cv2.imread(os.path.join(root_dir, fg))
    background = cv2.imread(os.path.join(root_dir, bg))
    combined = combine_fg_bg(foreground, background)

    # # Optionally, display the result
    # cv2.imshow('Output', combined)
    # cv2.waitKey(0)

    # Save the final image
    cv2.imwrite(os.path.join(output_dir, bg.replace("_bg.png", "_raw.png")), combined)
cv2.destroyAllWindows()