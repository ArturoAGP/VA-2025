import numpy as np
import cv2
img = cv2.imread(r"C:\Users\Arturo Alejandro\Documents\GitHub\VA-2025\001_p2_ImageOperations\watch (1).jpg")
img[55,55] = [255,255,255]
px = img[55,55]
img[100:150,100:150] = [255,255,255]
watch_face = img[37:111,107:194]
img[0:74,0:87] = watch_face

cv2.imshow('image',img)
cv2.waitKey(0)
