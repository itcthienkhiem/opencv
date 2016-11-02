import cv2
import numpy as np

img_orig = cv2.imread('../images/sudoku.jpg',0)
img = cv2.medianBlur(img_orig,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

# Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(img_orig,(5,5),0)
ret4,th4 = cv2.threshold(img_orig,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# contours
img_orig2 = cv2.imread('../images/sudoku.jpg')
contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        cnt_max = cnt
print(max_area)
print(cnt_max)

cv2.drawContours(img_orig2, cnt_max, -1, (0,255,0), 1)
cv2.imshow('contours', img_orig2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
            'otsu']
images = [img, th1, th2, th3, th4]

for i in range(len(images)):
    cv2.imshow(titles[i], images[i])


cv2.waitKey(0)
cv2.destroyAllWindows()
