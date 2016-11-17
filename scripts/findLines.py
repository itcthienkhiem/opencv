import cv2
import numpy as np

def main():
    image = cv2.imread('../images/lines_test.png', 0)
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
    cv2.bitwise_not(thresh, thresh)

    horizontal = thresh.copy()
    vertical = thresh.copy()

    horizontalsize = horizontal.shape[1] / 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    cv2.imshow('horizontal', horizontal)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
