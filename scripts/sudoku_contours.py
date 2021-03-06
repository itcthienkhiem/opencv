import cv2
import numpy as np
import math

def drawLines(lines, image, color=64):
    for line in lines:
        if (line[1]!=0):
            m = -1/math.tan(line[1])
            c = line[0]/math.sin(line[1])
            cv2.line(image,(0,int(c)),(image.shape[1],int(m*image.shape[1]+c)),color,1)
        else:
            cv2.line(image,(int(line[0]),0),(int(line[0]),image.shape[0]),color,1)

def mergeRelatedLines(lines, image):
    for i1, line in enumerate(lines):
        if(line[0]==0 and line[1]==-100):
            continue

        p1 = line[0]
        theta1 = line[1]

        if (theta1 > np.pi*45/180 and theta1 < np.pi*135/180):
            pt1current = [0, p1/math.sin(theta1)]
            pt2current = [image.shape[1], -image.shape[1]/math.tan(theta1)+p1/math.sin(theta1)]
        else:
            try: #????????
                pt1current = [p1/math.cos(theta1), 0]
                pt2current = [-image.shape[0]/math.tan(theta1)+p1/math.cos(theta1), image.shape[0]]
            except:
                pass

        for i2, line2 in enumerate(lines):
            if (i1 == i2):
                continue

            if (abs(line2[0]-line[0])<20 and abs(line2[1]-line[1])<np.pi*10/180):
                p = line2[0];
                theta = line2[1]

                if (theta > np.pi*45/180 and theta < np.pi*135/180):
                    pt1 = [0, p/math.sin(theta)]
                    pt2 = [image.shape[1], -image.shape[1]/math.tan(theta)+p/math.sin(theta)]
                else:
                    try:
                        pt1 = [p/math.cos(theta), 0]
                        pt2 = [-image.shape[0]/math.tan(theta)+p/math.cos(theta), image.shape[0]]
                    except:
                        pass

                if  (pt1[0]-pt1current[0])**2 + (pt1[1]-pt1current[1])**2 < 200 and \
                    (pt2[0]-pt2current[0])**2 + (pt2[1]-pt2current[1])**2 < 200 :

                    lines[i1][0] = (line[0]+line2[0])/2
                    lines[i1][1] = (line[1]+line2[1])/2

                    lines[i2][0] = 0
                    lines[i2][1] = -100


def main():
    image = cv2.imread('../images/sudoku.jpg', 0)
    blur = cv2.GaussianBlur(image,(11,11),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
    #cv2.bitwise_not(thresh, thresh)

    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    dilate = cv2.dilate(thresh,kernel)

    cv2.imshow('threshold', thresh)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area = -1
    cnt_max = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            cnt_max = cnt
    print(max_area)
    cv2.drawContours(image, cnt_max, -1, (0,255,0), 3)

    #grid = cv2.erode(dilate, kernel)

    #edges = cv2.Canny(dilate,100,200, L2gradient = True)
    #cv2.imshow('edges', edges)
    ''' trovare solo i contorni esterni '''
    #lines = cv2.HoughLines(grid,1,np.pi/180,150)

    ''' migliorare funzione di merge '''
    #mergeRelatedLines(lines[0], image)
    #drawLines(lines[0], grid)

    ''' trovare intersezione per trovare angoli '''
    ''' ampliare immagine se angoli esterni '''
    ''' adattare griglia '''
    ''' trovare tutte le righe/colonne -> celle '''
    ''' verificare se ci sono pedine/numeri -> identificarli '''

    cv2.imshow('source', image)
    #cv2.imshow('grid', grid)

    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'):
        cv2.imwrite('../images/grid.png',grid)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
