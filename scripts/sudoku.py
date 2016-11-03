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
                try:
                    if (theta > np.pi*45/180 and theta < np.pi*135/180):
                        pt1 = [0, p/math.sin(theta)]
                        pt2 = [image.shape[1], -image.shape[1]/math.tan(theta)+p/math.sin(theta)]
                    else:
                        pt1 = [p/math.cos(theta), 0]
                        pt2 = [-image.shape[0]/math.tan(theta)+p/math.cos(theta), image.shape[0]]

                    if  (pt1[0]-pt1current[0])**2 + (pt1[1]-pt1current[1])**2 < 200 and \
                        (pt2[0]-pt2current[0])**2 + (pt2[1]-pt2current[1])**2 < 200 :

                        lines[i1][0] = (line[0]+line2[0])/2
                        lines[i1][1] = (line[1]+line2[1])/2

                        lines[i2][0] = 0
                        lines[i2][1] = -100
                except:
                    pass

def main(image_src='sudoku.jpg'):
    image = cv2.imread('../images/%s'%image_src, 0)
    blur = cv2.GaussianBlur(image,(11,11),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
    cv2.bitwise_not(thresh, thresh)

    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    dilateMain = cv2.dilate(thresh,kernel)

    height, width = dilateMain.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)

    ''' provare con il flood maggiore, se non corrisponde ad una provare con
        il secondo maggiore e cosi via '''

    dilate = dilateMain.copy()
    max_dim = max(height, width)

    max_area = -1
    for i in range(max_dim):
        r = i%height
        c = i%width
        if dilate[r, c] == 255:
            area, rect = cv2.floodFill(dilate, mask, (c,r), 64) # changes on mask
            if area > max_area:
                max_area = area
                point = (c,r)

        c = width-1-c
        if dilate[r, c] == 255:
            area, rect = cv2.floodFill(dilate, mask, (c,r), 64) # changes on mask
            if area > max_area:
                max_area = area
                point = (c,r)

    ''' provare a trovare il flood maggiore solo per diagonali o anche mediane '''

    mask = np.zeros((height+2, width+2), np.uint8) # come funziona mask????
    area, rect = cv2.floodFill(dilate, mask, point, 255)

    for r in range(height):
        for c in range(width):
            if dilate[r, c] == 64:
                cv2.floodFill(dilate, mask, (c,r), 0)
            if dilate[r, c] == 255:
                if point[0] == c and point[1] == r:
                    continue
                cv2.floodFill(dilate, mask, (c,r), 0)


    grid = cv2.erode(dilate, kernel)

    #edges = cv2.Canny(dilate,100,200, L2gradient = True)
    #cv2.imshow('edges', edges)

    contours, hierarchy = cv2.findContours(grid.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt_max = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(cnt_max, True)
    print(perimeter)
    approx = cv2.approxPolyDP(cnt_max, 0.04 * perimeter, True)
    print(approx)
    print(len(approx))

    cv2.drawContours(mask, cnt_max, -1, 255, 1)
    for point in approx:
        point = point[0]
        cv2.circle(mask, (point[0],point[1]),   8, 255, -1)

    # corners = cv2.goodFeaturesToTrack(mask,4,0.1,10)
    # corners = np.int0(corners)
    #
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(mask,(x,y),5,255,-1)
    #
    # rect = cv2.minAreaRect(cnt_max) # (centox, centroy), (w,h), angolo
    # print(rect)
    # box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
    # print(box)
    # box = np.int0(box)
    # cv2.drawContours(mask,[box],0,64,2)

    # rect_w, rect_h = map(int, rect[1])
    # l = max(rect_w, rect_h)
    #
    # pts1 = np.float32([box[1], box[2], box[0], box[3]])
    # pts2 = np.float32([[0,0], [l-1, 0], [0,l-1], [l-1, l-1]])
    # M = cv2.getPerspectiveTransform(pts1,pts2)
    # dst = cv2.warpPerspective(grid,M,(l,l))
    # cv2.imshow('distors', dst)

    # # trovo punti estremi (angoli) del contorno
    # extLeft = tuple(cnt_max[cnt_max[:, :, 0].argmin()][0])
    # extRight = tuple(cnt_max[cnt_max[:, :, 0].argmax()][0])
    # extTop = tuple(cnt_max[cnt_max[:, :, 1].argmin()][0])
    # extBot = tuple(cnt_max[cnt_max[:, :, 1].argmax()][0])
    # #cv2.drawContours(mask, cnt_max, -1, 255, 3)
    #
    # cv2.circle(grid, extLeft,   8, 255, -1)
    # cv2.circle(grid, extRight,  8, 255, -1)
    # cv2.circle(grid, extTop,    8, 255, -1)
    # cv2.circle(grid, extBot,    8, 255, -1)
    ''' trovare solo i contorni esterni '''

    # lines = cv2.HoughLines(grid,1,np.pi/180,200)
    #
    # ''' migliorare funzione di merge '''
    # mergeRelatedLines(lines[0], image)
    # drawLines(lines[0], mask)

    #cv2.circle(grid,(rect[0], rect[3]), 10, 255, -1)
    #cv2.circle(grid,(rect[2], rect[1]), 10, 255, -1)



    ''' trovare intersezione per trovare angoli '''
    ''' ampliare immagine se angoli esterni '''
    ''' adattare griglia '''
    ''' trovare tutte le righe/colonne -> celle '''
    ''' verificare se ci sono pedine/numeri -> identificarli '''

    cv2.imshow('source', image)
    cv2.imshow('threshold', thresh)
    cv2.imshow('mask', mask)
    cv2.imshow('grid', grid)

    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'):
        cv2.imwrite('../images/grid.png',grid)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
