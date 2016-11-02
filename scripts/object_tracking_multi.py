import cv2
import numpy as np
import rgb2hsv

cap = cv2.VideoCapture(0)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

lower_green = np.array(rgb2hsv.convert([0,52,0], rgb2hsv.OPENCV_CONVERSION))
upper_green = np.array(rgb2hsv.convert([0,255,0], rgb2hsv.OPENCV_CONVERSION))

lower_red = np.array(rgb2hsv.convert([113,0,0], rgb2hsv.OPENCV_CONVERSION))
upper_red = np.array(rgb2hsv.convert([255,0,0], rgb2hsv.OPENCV_CONVERSION))

print(lower_red)
print(upper_red)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res1 = cv2.bitwise_and(frame,frame, mask= mask1)

    # mask2 = cv2.inRange(hsv, lower_green, upper_green)
    # res2 = cv2.bitwise_and(frame,frame, mask= mask2)
    #
    # mask3 = cv2.inRange(hsv, lower_red, upper_red)
    # res3 = cv2.bitwise_and(frame,frame, mask= mask3)

    cv2.imshow('frame',frame)
    cv2.imshow('mask1',mask1)
    # cv2.imshow('mask2',mask1)
    # cv2.imshow('mask3',mask1)
    cv2.imshow('res',res1)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
