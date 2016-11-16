import cv2
import numpy as np
import math

def drawLines(lines, image, color=64):
    for line in lines:
        if (line[1]!=0):
            m = -1/math.tan(line[1])
            c = line[0]/math.sin(line[1])
            # print("%d %d"%(0,int(c)))
            # print("%d %d"%(image.shape[1],int(m*image.shape[1]+c)))
            cv2.line(image,(0,int(c)),(image.shape[1],int(m*image.shape[1]+c)),color,1)
        else:
            cv2.line(image,(int(line[0]),0),(int(line[0]),image.shape[0]),color,1)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def main(image_src='sudoku.jpg'):
    image = cv2.imread('../images/%s'%image_src, 0)
    blur = cv2.GaussianBlur(image,(11,11),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
    cv2.bitwise_not(thresh, thresh)

    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    dilate = cv2.dilate(thresh,kernel)
    cpDilate = dilate.copy()

    height, width = dilate.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)

    ''' provare con il flood maggiore, se non corrisponde ad una provare con
        il secondo maggiore e cosi via '''

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

    contours, hierarchy = cv2.findContours(grid.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    cnt_max = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(cnt_max, True)
    print(perimeter)
    approx = cv2.approxPolyDP(cnt_max, 0.04 * perimeter, True)
    print(approx)
    print(len(approx))

    cv2.drawContours(mask, cnt_max, -1, 255, 1)

    vertex = list()
    for point in approx:
        point = point[0]
        vertex.append(point)
        cv2.circle(mask, (point[0],point[1]),   8, 255, -1)

    # (x, y, w, h) = cv2.boundingRect(approx)


    # corners = cv2.goodFeaturesToTrack(mask,4,0.1,10)
    # corners = np.int0(corners)
    #
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(mask,(x,y),5,255,-1)

    # rect = cv2.minAreaRect(cnt_max) # (centrox, centroy), (w,h), angolo
    # print(rect)
    # box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
    # # print(box)
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

    # edges = cv2.Canny(grid,50,150,apertureSize = 3)
    # cv2.imshow('edges', edges)

    # lines = cv2.HoughLines(grid,1,np.pi/180,200)

    ''' migliorare funzione di merge '''
    # mergeRelatedLines(lines[0], image)
    # drawLines(lines[0], mask)

    #cv2.circle(grid,(rect[0], rect[3]), 10, 255, -1)
    #cv2.circle(grid,(rect[2], rect[1]), 10, 255, -1)

    # vedi fitLine !!!!!!!!!!!!!!!


    #if vertex[0][0] > width/2:
        # il punto si trova sulla parte destra dell'immagine

    pts = np.array(vertex, dtype = "float32")
    dst = four_point_transform(thresh, pts) # thresh o cpDilate
    cv2.imshow('dst',dst)


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
        for img_path in sys.argv[1:]:
            main(img_path)
    else:
        main()

'''
documento:
    - obbiettivo
    - dove funziona
    - problematiche
    - opencv e funzioni utilizzate
    - interfaccia (libreria + linea di comando / interfaccia grafica)
    - esempi
    - codice
'''
