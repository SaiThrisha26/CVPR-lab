import cv2
import numpy as np
import matplotlib.pyplot as plt

# Histogram specification function
def hspe(inp, re):
    # Compute histograms
    hin = cv2.calcHist([inp], [0], None, [256], [0, 256])
    hre = cv2.calcHist([re], [0], None, [256], [0, 256])

    # Compute cumulative distribution functions (CDFs)
    cdfin = hin.cumsum() / hin.sum()
    cdfre = hre.cumsum() / hre.sum()

    # Create mapping function
    mapping = np.interp(cdfin, cdfre, np.arange(256))

    # Apply transformation
    out = mapping[inp]  # Map input pixels using computed mapping
    return np.uint8(out)

# Read the images
inp = cv2.imread("D:\CVPR\cvpr lab\m1.jpeg", cv2.IMREAD_GRAYSCALE)
re = cv2.imread("D:\CVPR\cvpr lab\m2.jpg", cv2.IMREAD_GRAYSCALE)

# ✅ Check if images are loaded properly
if inp is None or re is None:
    print("Error: One or both images could not be loaded. Check the file paths.")
    exit()

# Apply histogram specification
out = hspe(inp, re)

# Compute histograms for visualization
hist_inp = cv2.calcHist([inp], [0], None, [256], [0, 256])
hist_re = cv2.calcHist([re], [0], None, [256], [0, 256])
hist_out = cv2.calcHist([out], [0], None, [256], [0, 256])

# Plot images and histograms
plt.figure(figsize=(12, 6))

# Input Image
plt.subplot(2, 3, 1)
plt.imshow(inp, cmap="gray")
plt.title("Input Image")
plt.axis("off")

# Input Histogram
plt.subplot(2, 3, 2)
plt.plot(hist_inp, color="black")
plt.title("Input Histogram")

# Reference Image
plt.subplot(2, 3, 3)
plt.imshow(re, cmap="gray")
plt.title("Reference Image")
plt.axis("off")

# Reference Histogram
plt.subplot(2, 3, 4)
plt.plot(hist_re, color="black")
plt.title("Reference Histogram")

# Output Image
plt.subplot(2, 3, 5)
plt.imshow(out, cmap="gray")
plt.title("Transformed Image")
plt.axis("off")

# Output Histogram
plt.subplot(2, 3, 6)
plt.plot(hist_out, color="black")
plt.title("Transformed Histogram")

plt.tight_layout()
plt.show()
