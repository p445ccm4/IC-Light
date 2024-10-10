import cv2

# Load the image
image = cv2.imread("/home/michaelch/IC-Light/outputs/09102024/3_3_result.png")

# Apply Gaussian blur
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

# Save or display the result
cv2.imshow("smoothed_image.png", smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()