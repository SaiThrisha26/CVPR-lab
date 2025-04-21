import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_hit_or_miss

# Define a binary image as a NumPy array
image = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

# Define a structuring element (SE)
struct_element = np.array([
    [-1,  1, -1],
    [-1,  1, -1],
    [-1,  1, -1]
])


# Apply Hit-or-Miss transformation
hit_miss_result = binary_hit_or_miss(image, 
                                     structure1=(struct_element == 1), 
                                     structure2=(struct_element == 0))

# Convert result to 255 (white) for better visualization
hit_miss_result = hit_miss_result.astype(np.uint8) * 255

# Display Results
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(hit_miss_result, cmap='gray', vmin=0, vmax=255)
plt.title("Hit-or-Miss Output")
plt.axis('off')

plt.tight_layout()
plt.show()
