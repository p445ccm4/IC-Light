import os

import cv2
import numpy as np


def combine_fg_bg(foreground, background, result, shadow):
    # Create a mask for the grey background (RGB: 127, 127, 127)
    grey_background = np.array([127, 127, 127], dtype=np.uint8)
    mask = cv2.inRange(foreground, grey_background, grey_background)

    # Invert the mask to get the object (not grey)
    mask_inv = cv2.bitwise_not(mask)

    # Smooth the mask using Erosion
    # kernel = np.ones((3, 3), np.uint8)
    # mask_inv = cv2.erode(mask_inv, kernel, iterations=1)  # Adjust kernel size for feathering

    # Get the object part from the foreground
    object_part = cv2.bitwise_and(result, result, mask=mask_inv)

    # Smooth the mask using Dilation
    # object_part = cv2.dilate(object_part, kernel, iterations=1)  # Adjust kernel size for feathering

    # Prepare the background by resizing it to match the foreground dimensions
    background_resized = cv2.resize(background, (result.shape[1], result.shape[0]))

    # Get the background part where the object will be placed
    background_part = cv2.bitwise_and(background_resized, background_resized, mask=mask)

    # Combine the object part and the background part
    combined = cv2.add(object_part, background_part)

    combined = combined.astype(np.float32) - shadow.astype(np.float32)

    return np.clip(combined, 0, 255).astype(np.uint8)

# Load images
root_dir = "../outputs/10102024/separated"
output_dir = "../outputs/10102024/combined"
img_paths = os.listdir(root_dir)
fg_paths = sorted(path for path in img_paths if path.endswith("_cropped_fg.png"))
bg_paths = sorted(path for path in img_paths if path.endswith("_bg.png"))
result_paths = sorted(path for path in img_paths if path.endswith("_result.png"))
shadow_paths = sorted(path for path in img_paths if path.endswith("_shadow.png"))

for fg, bg, r, s in zip(fg_paths, bg_paths, result_paths, shadow_paths):
    foreground = cv2.imread(os.path.join(root_dir, fg))
    background = cv2.imread(os.path.join(root_dir, bg))
    result = cv2.imread(os.path.join(root_dir, r))
    shadow = cv2.imread(os.path.join(root_dir, s))
    combined = combine_fg_bg(foreground, background, result, shadow)

    # # Optionally, display the result
    # cv2.imshow('Output', combined)
    # cv2.waitKey(0)

    # Save the final image
    cv2.imwrite(os.path.join(output_dir, bg.replace("_bg.png", "_result.png")), combined)
cv2.destroyAllWindows()