# Write a program to implement histogram calculation and equalization for the given image. 
import cv2
import matplotlib.pyplot as plt
img=cv2.imread(r"5d0\5d0\apple image.jpg",cv2.IMREAD_GRAYSCALE)
hist=cv2.calcHist([img],[0],None,[255],[0,255])
eq=cv2.equalizeHist(img)
eqhist=cv2.calcHist([eq],[0],None,[255],[0,255])
plt.subplot(2,2,1),plt.imshow(img,cmap="gray"),plt.title("img"),plt.axis("off")
plt.subplot(2,2,2),plt.imshow(eq,cmap="gray"),plt.title("eq img"),plt.axis("off")
plt.subplot(2,2,3),plt.plot(hist,color="black"),plt.title("img hist")
plt.subplot(2,2,4),plt.plot(eqhist,color="black"),plt.title("eq hist")

plt.tight_layout()
plt.show()
