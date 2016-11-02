#!/usr/lib/python2.7

import numpy as np
import cv2

def main():
    # Load an color image in grayscale
    img1 = cv2.imread('expanse.jpg',0)  #cv2.IMREAD_GRAYSCALE
    img2 = cv2.imread('expanse.jpg',1)  #cv2.IMREAD_COLOR
    img3 = cv2.imread('expanse.jpg',-1) #cv2.IMREAD_UNCHANGED

    cv2.imshow('image1',img1)
    cv2.imshow('image2',img2)

    cv2.namedWindow('image3', cv2.cv.CV_WINDOW_NORMAL) # resizable
    cv2.imshow('image3',img3)

    while 1:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('expanse_gray.png',img1)
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
