import cv2
import numpy as np

# Load an image in grayscale (binary image)
image = cv2.imread('apple image.jpg', cv2.IMREAD_GRAYSCALE)

# Create a structuring element (kernel)
kernel = np.ones((15, 15), np.uint8)  # A 5x5 square kernel

# 1. Erosion to shrink more add more iterations
eroded_image = cv2.erode(image, kernel, iterations=1)
# eroded_image1= cv2.erode(image, kernel, iterations=5)
# 2. Dilation to expand more add more iterations
dilated_image = cv2.dilate(image, kernel, iterations=1)

# 3. Opening (erosion followed by dilation)
opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 4. Closing (dilation followed by erosion)
closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Eroded Image', eroded_image)
# cv2.imshow('Eroded Image1', eroded_image1)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Opened Image', opened_image)
cv2.imshow('Closed Image', closed_image)

# Wait until a key is pressed and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
