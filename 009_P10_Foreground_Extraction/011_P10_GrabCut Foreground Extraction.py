import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
img2 = cv2.imread('opencv-corner-detection-sample.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
corners = np.int16(corners)
rect = (50,50,700,1200)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
for corner in corners:
 x,y = corner.ravel()
 cv2.circle(img2,(x,y),3,255,-1)
cv2.imshow('Corner',img2)
plt.imshow(img)
plt.colorbar()
plt.show()
