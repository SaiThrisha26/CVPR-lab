import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('apple image.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error loading image!")
    exit()

# 1. Sobel Edge Detection (X and Y gradients)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)

# 2. Prewitt Edge Detection using custom kernels
prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
prewitt_x_edges = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
prewitt_y_edges = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
prewitt_edges = cv2.magnitude(prewitt_x_edges, prewitt_y_edges)

# 3. Laplacian Edge Detection
laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)

# 4. Canny Edge Detection
# Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
canny_edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

# Display results using matplotlib
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.axis('off')

# Sobel Edge Detection
plt.subplot(2, 3, 2), plt.imshow(sobel_edges, cmap='gray'), plt.title('Sobel Edges')
plt.axis('off')

# Prewitt Edge Detection
plt.subplot(2, 3, 3), plt.imshow(prewitt_edges, cmap='gray'), plt.title('Prewitt Edges')
plt.axis('off')

# Laplacian Edge Detection
plt.subplot(2, 3, 4), plt.imshow(laplacian_edges, cmap='gray'), plt.title('Laplacian Edges')
plt.axis('off')

# Canny Edge Detection
plt.subplot(2, 3, 5), plt.imshow(canny_edges, cmap='gray'), plt.title('Canny Edges')
plt.axis('off')

# Show all the images in a grid
plt.tight_layout()
plt.show()
