import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load the image
image = cv2.imread(r'D:\CVPR\cvpr lab\5d0\5d0\apple image.jpg',cv2.IMREAD_COLOR_RGB)  
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB (for matplotlib)

# Step 2: Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)  # Reshape to (num_pixels, 3) for RGB color channels

# Step 3: Apply K-Means clustering
K = 3  # Number of clusters (segments)
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(pixels)

# Step 4: Replace each pixel with its corresponding cluster center
segmented_image = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_image.reshape(image.shape)  # Reshape back to the original image shape

# Step 5: Display the original and segmented images
#plt.figure(figsize=(10, 5))

# Display original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Display segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmented_image.astype(int))  # Convert to int for proper display
plt.title('Segmented Image with K-Means')
plt.axis('off')

plt.show()
