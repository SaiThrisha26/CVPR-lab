# Enhance a grayscale image using three techniques: negative, logarithmic, and gamma (power-law) transformations.
import cv2
import numpy as np
import matplotlib.pyplot as plt
def show(img,t):
    plt.figure()
    plt.imshow(img,cmap='gray')
    plt.title(t)
    plt.axis("off")
img=cv2.imread(r"5d0\5d0\apple image.jpg",cv2.IMREAD_GRAYSCALE)
neg=255-img
show(neg,"neg")
def logt(img):
    ep=1e-5
    imgf=img.astype(np.float32)+ep
    maxv=np.max(imgf)
    if maxv==0:
        maxv=1
    c=255/np.log(maxv+1)
    imgt=c*np.log(1+imgf)
    return np.uint8(imgt)
show(logt(img),"logt")
gamma=[0.2,0.5,1,2,5]
for g in gamma:
    gimg=255*((img/255)**g)
    show(gimg,f"gamma:{g}")
    
