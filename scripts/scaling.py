import cv2
import numpy as np

img = cv2.imread('E:\opencv\messi5.jpg')

res1 = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#OR

height, width = img.shape[:2]
res2 = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

cv2.imshow('res1',res1)
cv2.imshow('res2',res2)
cv2.waitKey(0)
