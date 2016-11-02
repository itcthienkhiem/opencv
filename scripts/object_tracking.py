import cv2
import numpy as np
import rgb2hsv

cv2.namedWindow('image', cv2.cv.CV_WINDOW_NORMAL)

# create trackbars for color change
cv2.createTrackbar('R1','image',0,255,lambda x: None)
cv2.createTrackbar('R2','image',0,255,lambda x: None)
cv2.createTrackbar('G1','image',0,255,lambda x: None)
cv2.createTrackbar('G2','image',0,255,lambda x: None)
cv2.createTrackbar('B1','image',0,255,lambda x: None)
cv2.createTrackbar('B2','image',0,255,lambda x: None)

#cap = cv2.VideoCapture(0)
frame = cv2.imread('../images/opencv_logo.jpg')
b1 = b2 = g1 = g2 = r1 = r2 = 0
while(1):

    # Take each frame
#    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])

    #lower = rgb2hsv.convert([r1, g1, b1], rgb2hsv.OPENCV_CONVERSION)
    #upper = rgb2hsv.convert([r2, g2, b2], rgb2hsv.OPENCV_CONVERSION)
    lower = np.array([b1,g1,r1])
    upper = np.array([b2,g2,r2])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('image',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r1 = cv2.getTrackbarPos('R1','image')
    r2 = cv2.getTrackbarPos('R2','image')
    g1 = cv2.getTrackbarPos('G1','image')
    g2 = cv2.getTrackbarPos('G2','image')
    b1 = cv2.getTrackbarPos('B1','image')
    b2 = cv2.getTrackbarPos('B2','image')

cv2.destroyAllWindows()
