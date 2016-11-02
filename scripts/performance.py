
import cv2
import time

img = cv2.imread("messi5.jpg")

cv2.setUseOptimized(True)
print("optimized: %s"%cv2.useOptimized())

t1 = time.time()
for i in range(5,49,2):
    img = cv2.medianBlur(img,i)
t2 = time.time()

print("tot time: %s"%(t2-t1))


cv2.setUseOptimized(False)
print("optimized: %s"%cv2.useOptimized())

t1 = time.time()
for i in range(5,49,2):
    img = cv2.medianBlur(img,i)
t2 = time.time()

print("tot time: %s"%(t2-t1))
