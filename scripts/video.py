import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(cap.isOpened()): # if not opened -> cap.open()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret: continue

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): # non bloccante waitKey(n) con n>0
        break
    elif k == ord('s'):
        cv2.imwrite

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
