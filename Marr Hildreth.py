import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread(r'D:\CVPR\cvpr lab\5d0\5d0\apple image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(image, (7, 7), 1.5)

# Compute Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# Detect Zero Crossings properly
zero_crossing = np.zeros_like(laplacian, dtype=np.uint8)
laplacian_shifted = np.roll(laplacian, 1, axis=0)  # shift down
zero_crossing[np.where((laplacian * laplacian_shifted) < 0)] = 255

laplacian_shifted = np.roll(laplacian, 1, axis=1)  # shift right
zero_crossing[np.where((laplacian * laplacian_shifted) < 0)] = 255

# Display
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(zero_crossing, cmap='gray')
plt.title("Marr-Hildreth Edges")
plt.axis('off')

plt.tight_layout()
plt.show()
