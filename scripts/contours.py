import cv2
import numpy as np
from matplotlib import pyplot as plt

img_orig = cv2.imread('..\images\sudoku.jpg')
imgray = cv2.cvtColor(img_orig,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,100,255,0)


contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image_contours = cv2.drawContours(img_orig, contours, -1, (0,255,0), 1)

cv2.imshow('original', img_orig)
cv2.imshow('thresholding', thresh)
cv2.imshow('contours', img_orig)

cv2.waitKey(0)
cv2.destroyAllWindows()
