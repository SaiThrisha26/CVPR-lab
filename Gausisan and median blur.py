import cv2
import numpy as np

# Load an image
image = cv2.imread('apple image.jpg')

# Gaussian Filter
# Apply GaussianBlur with a larger kernel (15x15) and a higher standard deviation (5)
gaussian_blurred = cv2.GaussianBlur(image, (15, 15), 5)

# Median Filter
# Apply medianBlur with a larger kernel size (15)
median_blurred = cv2.medianBlur(image, 15)

# Display images
cv2.imshow("Original", image)
cv2.imshow("Gaussian Blurred", gaussian_blurred)
cv2.imshow("Median Blurred", median_blurred)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
