import cv2
import numpy as np

img = cv2.imread('../images/j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

cv2.imshow('orig', img)
cv2.imshow('erode', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
