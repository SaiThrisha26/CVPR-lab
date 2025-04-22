import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread(r'D:\CVPR\cvpr lab\5d0\5d0\nature.jpg', cv2.IMREAD_GRAYSCALE)  # Make sure the image is in the same folder

# Apply PCA treating each row as a sample
pca = PCA(n_components=50)#less components means more blur
reduced_image = pca.fit_transform(image)

# Reconstruct the image
reconstructed_image = pca.inverse_transform(reduced_image)

# Clip values to be valid pixel range and convert to uint8
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# Display using matplotlib
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image (PCA)')
plt.axis('off')

plt.tight_layout()
plt.show()
