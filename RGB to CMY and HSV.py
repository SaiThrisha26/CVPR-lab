import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (BGR format)
image_bgr = cv2.imread(r'D:\CVPR\cvpr lab\5d0\5d0\apple image.jpg')

# Check if image loaded properly
if image_bgr is None:
    print("Error: Image not found.")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert RGB to CMY
image_cmy = 1 - (image_rgb / 255.0)
image_cmy = (image_cmy * 255).astype(np.uint8)

# Convert BGR (original) to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Display the images
cv2.imshow('Original RGB Image', image_rgb)  # convert back to BGR for imshow
cv2.imshow('CMY Image',image_cmy)           # convert back to BGR for imshow
cv2.imshow('HSV Image', image_hsv)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
