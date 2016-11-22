import cv2
import numpy as np


image = cv2.imread("../images/sudoku.jpg", 0)
blur = cv2.GaussianBlur(image,(11,11),0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,2)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
canvas = np.zeros(thresh.shape[:2], np.uint8)
mask = np.zeros(map(lambda x: x+2, thresh.shape[:2]), np.uint8)

for cnt in contours:
    cv2.drawContours(canvas, cnt, -1, 255, thickness=cv2.cv.CV_FILLED)

cv2.imshow('sdd', canvas)
cv2.waitKey(0)

cv2.destroyAllWindows()


    # for i_cnt in bestGroup[0]+bugs+[0]:
    #     tl, tr, br, bl, cnt = verticalLines[i_cnt]
    #     dst = np.array([tl, tr, br, bl], dtype = "float32")
    #
    # 	M = cv2.getPerspectiveTransform(dst, dst)
    # 	line = cv2.warpPerspective(vertical, M, vertical.shape[:2])
