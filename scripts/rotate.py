import cv2
import numpy as np

img = cv2.imread('E:\opencv\messi5.jpg',0)
rows,cols = img.shape[:2]

#                          center,     angle, scale
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,float(rows)/cols)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
