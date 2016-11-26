import cv2
import numpy as np
import math

def findGrid(thresh):
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    dilate = cv2.dilate(thresh,kernel)

    height, width = dilate.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)

    max_dim = max(height, width)
    max_area = -1
    # trovo l'oggetto con area maggiore
    for i in range(max_dim):
        r = i%height
        c = i%width
        # diagonale discendente
        if dilate[r, c] == 255:
            area, rect = cv2.floodFill(dilate, mask, (c,r), 64)
            if area > max_area:
                max_area = area
                point = (c,r)

        c = width-1-c
        # diagonale ascendente
        if dilate[r, c] == 255:
            area, rect = cv2.floodFill(dilate, mask, (c,r), 64)
            if area > max_area:
                max_area = area
                point = (c,r)

    mask = np.zeros((height+2, width+2), np.uint8)
    area, rect = cv2.floodFill(dilate, mask, point, 255)

    for r in range(height):
        for c in range(width):
            if dilate[r, c] == 64:
                cv2.floodFill(dilate, mask, (c,r), 0)
            if dilate[r, c] == 255:
                if point[0] == c and point[1] == r:
                    continue
                cv2.floodFill(dilate, mask, (c,r), 0)

    return cv2.erode(dilate, kernel)

def mergeLines(lines, width, height):
    rows = list()
    cols = list()
    m_avg = 0
    nHorizLines = 0
    for line in lines:
        if (line[1]!=0): # theta
            m = -1/math.tan(line[1])
            c = line[0]/math.sin(line[1])
            # (0, c) (width, m*width+c)
            # x,y con origine in alto a sinistra
            # c = row[0]*(math.sin(row[1]) - m*math.cos(row[1])) # y = mx + q -> q = y - mx

            """
            Una volta raddrizzata piu o meno la griglia attraverso
            l approssimazione a un rettangolo posso considerare per le rette
            orizzontali l intercetta c come valore compreso tra 0 e height
            e il coefficiente angolare m molto piccolo.
            Posso invece considerare m molto grande e c molto elevata per le
            rette verticali. In questo modo elimino rette completamente sbagliate.
            """

            if m > m_avg-0.3 and m < m_avg+0.3: # rette orizzontali
                ''' creo dei gruppi secondo l'intercetta c '''
                groupFind = False
                for groupRow in rows:
                    c_avg, group = groupRow
                    if c < c_avg+height/20 and c > c_avg-height/20: # range in cui varia c
                        groupFind = True
                        group.append(line)
                        groupRow[0] = (c_avg*(len(group)-1)+c)/len(group)

                if not groupFind: rows.append([c, [line]])

                m_avg = (m_avg*nHorizLines+m)/float(nHorizLines+1)
                nHorizLines += 1

            elif abs(m) > 3: # minima inclinazione della retta verticale
                zero = -c/m
                ''' creo dei gruppi secondo l'intersezione con l'asse delle ascisse '''
                groupFind = False
                for groupCol in cols:
                    zero_avg, group = groupCol
                    if zero < zero_avg+width/20 and zero > zero_avg-width/20: # da stabilire il range valido
                        groupFind = True
                        group.append(line)
                        groupCol[0] = (zero_avg*(len(group)-1)+zero)/len(group)

                if not groupFind: cols.append([zero, [line]])

            else:
                print('oblique line')
        else:
            zero = line[0] # cioe la distanza dall'origine
            groupFind = False
            for groupCol in cols:
                zero_avg, group = groupCol
                if zero < zero_avg+width/20 and zero > zero_avg-width/20: # da stabilire il range valido
                    groupFind = True
                    group.append(line)
                    groupCol[0] = (zero_avg*(len(group)-1)+zero)/len(group)

            if not groupFind: cols.append([zero, [line]])

    print("rows: %d\ncols: %d"%(len(rows),len(cols)))

    ggRows = list()
    for c_avg, group in rows:
        diff = 100
        bestRow = None
        for row in group:
            c = row[0]/math.sin(row[1])
            if abs(c_avg - c) < diff:
                diff = abs(c_avg-c)
                bestRow = row
        ggRows.append(bestRow)

    ggCols = list()
    for groupCol in cols:
        zero_avg, group = groupCol

        diff = 100
        bestCol = None
        for col in group:
            if col[1]!=0:
                c = col[0]/math.sin(col[1])
                m = -1/math.tan(col[1])
                zero = -c/m
            else:
                zero = col[0]

            if abs(zero_avg - zero) < diff:
                diff = abs(zero_avg - zero)
                bestCol = col

        ggCols.append(bestCol)

    #                                        c                                       zero
    return sorted(ggRows, key= lambda r: r[0]/math.sin(r[1])), sorted(ggCols, key=calcZero)

def calcZero(line):
    try:
        m = -1/math.tan(line[1])
        c = line[0]/math.sin(line[1])
        return -c/m
    except:
        return line[0]

def method1(image, show_all, show):
    blur = cv2.GaussianBlur(image,(11,11),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,2)
    if show_all and show: cv2.imshow('threshold1', thresh)

    grid = findGrid(thresh) # find max flood object
    if show_all and show: cv2.imshow('grid1', grid)

    contours, hierarchy = cv2.findContours(grid.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # senza .copy() e meglio?
    cnt_max = max(contours, key=cv2.contourArea)

    # perimeter = cv2.arcLength(cnt_max, True)
    # approx = cv2.approxPolyDP(cnt_max, 0.04 * perimeter, True) # trovo approssimati punti del rettangolo

    # for point in approx:
    #     point = point[0]
    #     cv2.circle(canvas, (point[0],point[1]),   8, 255, -1)


    # corners = cv2.goodFeaturesToTrack(canvas,4,0.1,10)
    # corners = np.int0(corners)
    #
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(canvas,(x,y),5,255,-1)

    rect = cv2.minAreaRect(cnt_max) # (centrox, centroy), (w,h), angolo
    box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
    box = np.int0(box)
    # cv2.drawContours(canvas,[box],0,64,2)

    grid = four_point_transform(grid, box) # prendo e raddrizzo la porzione dell'immagine da elaborare

    lines = cv2.HoughLines(grid,1,np.pi/180,200)
    grid2 = grid.copy()
    drawLines(lines[0], grid2)
    if show_all: cv2.imshow('all lines', grid2)

    rr, cc = mergeLines(lines[0], *grid.shape[:2])
    drawLines(rr+cc, grid)
    cv2.imshow('grid', grid)

    return grid, rr, cc, box, cnt_max


def calcDistMiddlePoint(segment1, segment2):
    p1, p2 = segment1
    p3, p4 = segment2

    middlePoint = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

    if (p3[0] != p4[0]):
        m2 = (p3[1]-p4[1])/(p3[0]-p4[0])
        q2 = p3[1]-m2*p3[0]

        # distanza punto retta
        dist = abs(middlePoint[1] - (m2*middlePoint[0] + q2)) / math.sqrt(1+m2**2)
    else:
        # retta verticale
        dist = abs(p3[0]-middlePoint[0])

    return dist

def calcDistanceIndexCnt(i_cnt1, i_cnt2, verticalLines):
    tl1, tr, br, bl1, cnt = verticalLines[i_cnt1]
    tl2, tr, br, bl2, cnt = verticalLines[i_cnt2]
    return calcDistMiddlePoint((tl1, bl1), (tl2, bl2))

def drawFilledContour(cnt, image, mask, color=255, thickness=1):
    cv2.drawContours(image, cnt, -1, color, thickness)
    print(cnt)
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.floodFill(image, mask, (cX, cY), 255)


def method2(image, cnt_max, rect, show_all, show):
    blur = cv2.GaussianBlur(image,(9,9),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,2)
    if show_all: cv2.imshow('threshold2', thresh)

    height, width = thresh.shape[:2]
    gridMask = np.zeros((height, width), np.uint8)
    cv2.drawContours(gridMask, cnt_max, -1, 255, thickness=cv2.cv.CV_FILLED) # ???? perche non lo riempie??
    mask = np.zeros((height+2, width+2), np.uint8)

    tl, tr, br, bl = order_points(rect)
    cv2.floodFill(gridMask, mask, (int(br[0]+tl[0])/2, int(br[1]+tl[1])/2), 255) # !!!!!!!!!!! riempire tutto il contorno

    threshROI = cv2.bitwise_and(thresh, gridMask)
    threshROIAdapted = four_point_transform(threshROI, rect)
    # cv2.imshow('ROItransformed', threshROIAdapted)

    height, width = threshROIAdapted.shape[:2]
    h2 = np.zeros((height, width), np.uint8)
    v2 = np.zeros((height, width), np.uint8)

    horizontalsize = 20 # threshROIAdapted.shape[1] / 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

    expandeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    horizontal = cv2.morphologyEx(threshROIAdapted, cv2.MORPH_CLOSE, expandeKernel, iterations=1) # piu iterazioni se immagine distorta
    # horizontal = cv2.dilate(threshROIAdapted, expandeKernel) # espando un po' le linee orizzontali
    # horizontal = cv2.erode(threshROIAdapted, expandeKernel)

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    contours, hierarchy = cv2.findContours(horizontal.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    mask = np.zeros((h2.shape[0]+2, h2.shape[1]+2), np.uint8)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt) # (centrox, centroy), (w,h), angolo
        box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
        box = np.int0(box)
        tl, tr, br, bl = order_points(box)
        if (abs(tr[0]-tl[0]) > horizontal.shape[1]//4):
            cv2.drawContours(h2, cnt, -1, 255, thickness=-1)
            # !!!!!!!! da riempire il contorno per facilitare l'individuazione dei punti di intersezione
            # drawFilledContour(cnt, h2, mask, color=255, thickness=1)


    # cv2.imshow('horizontal', horizontal)

    # lines = cv2.HoughLines(horizontal,1,np.pi/180,200)
    # rr, cc = mergeLines(lines[0], *horizontal.shape[:2])
    # drawLines(rr, gridMask)

    verticalsize = 20 # threshROIAdapted.shape[0] / 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize))
    # verticalStructure = np.array([[0,0,1,0,0]]*5+[[0,1,1,1,0]]*5+[[1,1,0,1,1]]*5, np.uint8)

    expandeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    vertical = cv2.morphologyEx(threshROIAdapted, cv2.MORPH_CLOSE, expandeKernel, iterations=1)

    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # cv2.imshow('vertical', vertical)


    contours, hierarchy = cv2.findContours(vertical.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    verticalLines = list()
    for cnt in contours:
        rect = cv2.minAreaRect(cnt) # (centrox, centroy), (w,h), angolo
        box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
        box = np.int0(box)

        tl, tr, br, bl = order_points(box)
        if (abs(tl[1]-bl[1]) > vertical.shape[0]//4 and ((tl[0]-bl[0])**2+(tl[1]-bl[1])**2)/((bl[0]-br[0])**2+(bl[1]-br[1])**2) > 500):
            verticalLines.append([tl, tr, br, bl, cnt])


    verticalLines = sorted(verticalLines, key=lambda l: l[0][0])

    tl_prec, bl_prec = verticalLines[0][0], verticalLines[0][3]
    verticalDistances = list()
    for i, (tl, tr, br, bl, cnt) in enumerate(verticalLines[1:]):
        dist = calcDistMiddlePoint((tl, bl),(tl_prec, bl_prec))
        verticalDistances.append((1+i, dist))
        tl_prec, bl_prec = tl, bl

    # verticalDistances = sorted(verticalDistances, key=lambda dist: dist[1])
    print(verticalDistances)

    RANGE_DIST = vertical.shape[1]/80.0
    groups = list()
    temp = list()
    for i, dist in verticalDistances:
        if i in temp: continue
        group = [i]
        temp.append(i)
        distTot = dist
        for i2, dist2 in verticalDistances:
            if i==i2 or i2 in temp: continue

            if abs(dist-dist2) < RANGE_DIST:
                distTot += dist2
                group.append(i2)
                temp.append(i2)
        groups.append([group, float(distTot)/len(group)])

    print(groups)

    bestGroup = max(groups, key=lambda g: len(g[0]))
    distAvg = bestGroup[1]

    bugs = list()
    for i, i_cnt in enumerate(bestGroup[0][:-1]):
        i_cntNext = bestGroup[0][i+1]
        if i_cntNext - i_cnt == 1: continue

        for i, dist in verticalDistances[i_cnt:i_cntNext]:
            if distAvg-RANGE_DIST < calcDistanceIndexCnt(i, i_cnt, verticalLines) < distAvg+RANGE_DIST:
                bugs.append(i)
                i_cnt = i


    mask = np.zeros((v2.shape[0]+2, v2.shape[1]+2), np.uint8)
    for i_cnt in bestGroup[0]+bugs+[0]: # assumiamo che la prima riga sia effettivamente la prima !!! da sistemare
        cnt = verticalLines[i_cnt][-1]
        cv2.drawContours(v2, cnt, -1, 255, thickness=-1)
        # drawFilledContour(cnt, v2, mask, color=255, thickness=1)

    if show_all: cv2.imshow('h2',h2)
    if show_all: cv2.imshow('v2',v2)

    points = cv2.bitwise_and(h2,v2)
    # expandeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    # points = cv2.morphologyEx(points, cv2.MORPH_CLOSE, expandeKernel, iterations=1)

    contours, hierarchy = cv2.findContours(points.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    points = np.zeros(points.shape[:2], np.uint8)
    groups = list()
    for cnt in contours:
        x = 0
        y = 0
        for point in cnt:
            point = point[0]
            x += point[0]
            y += point[1]

        x, y = int(x/len(cnt)), int(y/len(cnt))

        groupFind = False
        for g in groups:
            if abs(g[-1][0]-x)<10 and abs(g[-1][1]-y)<10: # unisco i 4 punti
                g[-1] = [ (g[-1][0]*len(g[0])+x)/(len(g[0])+1), (g[-1][1]*len(g[0])+y)/(len(g[0])+1) ]
                g[0].append((x,y))
                groupFind = True
        if not groupFind:
            groups.append([[(x,y)], [x,y]])

    for g in groups:
        # print("%d %d"%(x,y))
        cv2.circle(points, tuple(g[-1]),   0, 255, -1)
    print("finded points: %d"%len(groups))

    cv2.imshow('intersezioni', points)
    # cv2.imshow('horizontal', horizontal)

    return groups, points

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
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

    # maximum width
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

    # maximum height
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

def RowColIntersection(row, col):
    m1 = -1/math.tan(row[1])
    c1 = row[0]/math.sin(row[1])

    if col[1] != 0:
        m2 = -1/math.tan(col[1])
        c2 = col[0]/math.sin(col[1])
        x = -(c1-c2) / (m1-m2)
    else:
        x = col[0]

    y = m1*x + c1
    return (x,y)

def calcCells(points, rr, cc):
    # calcolo corners teorici con intersezione linee
    # guardo se quella parte di immagine e presente, se c'e un punto vicino
    # se non presente provo su un altro spigolo
    # calcolo il punto orizz piu vicino, distanza massima di x
    # sia in su che in giu, se lo trovo lo contrassegno
    # calcolo il punto vert piu vicino, distanza massima di x
    # sia a dx che sx, se lo trovo lo contrassegno
    # rieseguo la funzione ricorsivamente sui punti appena trovati

    # points = np.int0(map(lambda x: x[-1], points))
    # corners = order_points(points)

    tl = RowColIntersection(rr[0], cc[0])
    tr = RowColIntersection(rr[0], cc[-1])
    br = RowColIntersection(rr[-1], cc[-1])
    bl = RowColIntersection(rr[-1], cc[0])

    corners = (tl, tr, br, bl)
    return corners


def main(image_src, only_m1, show_all, show=False):
    image = cv2.imread(image_src, 0)


    grid, rr, cc, rect, cnt_max = method1(image, show_all, show)
    # if not only_m1:
    points, pointImg = method2(image, cnt_max, rect, show_all, show)

    cells = calcCells(points, rr, cc)
    # for cell in cells:
    #     cv2.circle(pointImg, (cell[0], cell[1]),   8, 255, -1)
    # cv2.imshow('intersezioni', pointImg)

    # cv2.imshow('source', image)
    # cv2.imshow('gridMask', gridMask)
    # cv2.imshow('grid', grid)

    while 1:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite('../esempi/grid.png',grid)
            if not only_m1: cv2.imwrite('../esempi/points.png',pointImg)
            break
        elif k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='OpenCV. Recognise a grid and extract data cells.')

    parser.add_argument('img', metavar='img', nargs='?', default="../images/sudoku.jpg", help='image path')
    parser.add_argument('-a', '--show_all', dest="show_all", nargs='?', const=True, default=False, help='show all result images')
    parser.add_argument('-m1', '--method1', dest="only_m1", nargs='?', const=True, default=False, help='execute only method 1')

    args = parser.parse_args()

    main("../images/%s"%args.img, args.only_m1, args.show_all, show=True)
