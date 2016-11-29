import cv2
import numpy as np
from numpy.linalg import norm
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

    return thresh, grid, rr, cc, box, cnt_max


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

def getIntersectionPoint(intersection):
    contours,_ = cv2.findContours(intersection,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    x_avg = 0
    y_avg = 0
    for cnt in contours:
        x,y = cnt[0][0]
        x_avg += x
        y_avg += y
    x_avg /= len(contours)
    y_avg /= len(contours)

    return (x_avg, y_avg)

def method2(image, cnt_max, kernelTransform, show_all, show):
    blur = cv2.GaussianBlur(image,(9,9),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,2)
    if show_all: cv2.imshow('threshold2', thresh)

    height, width = thresh.shape[:2]
    gridMask = np.zeros((height, width), np.uint8)
    cv2.drawContours(gridMask, cnt_max, -1, 255, thickness=cv2.cv.CV_FILLED) # ???? perche non lo riempie??
    mask = np.zeros((height+2, width+2), np.uint8)

    tl, tr, br, bl = order_points(kernelTransform)
    cv2.floodFill(gridMask, mask, (int(br[0]+tl[0])/2, int(br[1]+tl[1])/2), 255) # !!!!!!!!!!! riempire tutto il contorno

    threshROI = cv2.bitwise_and(thresh, gridMask)
    threshROIAdapted = four_point_transform(threshROI, kernelTransform)
    if show_all: cv2.imshow('ROItransformed', threshROIAdapted)


    horizontalsize = 20 # threshROIAdapted.shape[1] / 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

    expandeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    horizontal = cv2.morphologyEx(threshROIAdapted, cv2.MORPH_CLOSE, expandeKernel, iterations=1) # piu iterazioni se immagine distorta
    # horizontal = cv2.dilate(threshROIAdapted, expandeKernel) # espando un po' le linee orizzontali
    # horizontal = cv2.erode(threshROIAdapted, expandeKernel)

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    contours, hierarchy = cv2.findContours(horizontal.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    rows = list()
    for cnt in contours:
        rect = cv2.minAreaRect(cnt) # (centrox, centroy), (w,h), angolo
        box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
        box = np.int0(box)
        tl, tr, br, bl = order_points(box)
        if (abs(tr[0]-tl[0]) > horizontal.shape[1]//4):
            rows.append((tl, tr, br, bl, cnt))
            # cv2.drawContours(h2, cnt, -1, 255, thickness=-1)
    rows = map( lambda r: r[-1],
                sorted(rows, key=lambda r: r[0][1])
              )

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


    cols = list()
    for i_cnt in bestGroup[0]+bugs+[0]: # assumiamo che la prima riga sia effettivamente la prima !!! da sistemare
        cnt = verticalLines[i_cnt][-1]

        rect = cv2.minAreaRect(cnt) # (centrox, centroy), (w,h), angolo
        box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
        box = np.int0(box)
        tl, tr, br, bl = order_points(box)
        cols.append((tl, tr, br, bl, cnt))
        # cv2.drawContours(v2, cnt, -1, 255, thickness=-1)
    cols = map( lambda c: c[-1],
                sorted(cols, key=lambda c: c[0][0])
              )

    height, width = threshROIAdapted.shape[:2]
    points = [[None for __ in range(len(cols))] for _ in range(len(rows))]
    canvas = np.zeros((height, width), np.uint8)

    for i_r, cnt_r in enumerate(rows):
        row_canvas = canvas.copy()
        cv2.drawContours(row_canvas, cnt_r, -1, 255, thickness=1)

        for i_c, cnt_c in enumerate(cols):
            col_canvas = canvas.copy()
            cv2.drawContours(col_canvas, cnt_c, -1, 255, thickness=1)

            intersection = cv2.bitwise_and(row_canvas, col_canvas)

            # ottengo il centro
            point = getIntersectionPoint(intersection)
            if point == None: print("Error: %d %d"%(i_r, i_c))
            points[i_r][i_c] = point

    pp_canvas = canvas.copy()
    for row_points in points:
        for p in row_points:
            cv2.circle(pp_canvas, p,   0, 255, -1)
        cv2.imshow("points", pp_canvas)
        # cv2.waitKey(1000)
    return points, pp_canvas


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


def extractCells(points, thresh, kernelTransform):
    threshAdapted = four_point_transform(thresh, kernelTransform)

    gridImages = [[None for __ in range(len(points[0]))] for _ in range(len(points))]

    for r, row_points in enumerate(points[:-1]):
        for c, point1 in enumerate(row_points[:-1]):
            point2 = points[r+1][c]
            point3 = points[r][c+1]
            point4 = points[r+1][c+1]
            if point1 and point2 and point3 and point4:
                cell = np.int0((point1, point2, point3, point4))
                imgCell = four_point_transform(threshAdapted, cell)
                gridImages[r][c] = imgCell
                # cv2.imshow("%d,%d"%(r,c), imgCell)

    return gridImages

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

def extractNumbers(cells):
    digits_img = cv2.imread("../images/digits.png", 0)
    h, w = digits_img.shape[:2]
    sx, sy = (20, 20) # cell dimensions
    digits = [np.hsplit(row, w//sx) for row in np.vsplit(digits_img, h//sy)]
    digits = np.array(digits)
    digits = digits.reshape(-1, sy, sx) # raggruppo righe

    labels = np.repeat(np.arange(10), len(digits)/10) # 10 number of images for the same digit

    digits2 = map(deskew, digits) # "raddrizza" l'immagine, il testo, le cifre
    samples = preprocess_hog(digits2)

    model = cv2.KNearest()
    model.train(samples, labels)

    cc = list()
    cc_ii = [[None for __ in range(len(cells[0]))] for _ in range(len(cells))]

    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            if cell is not None:
                cc_ii[r][c] = r*9 + c # !!!!!!!!!!
                #cv2.imshow("%d"%i, cell)
                cc.append(cell)

    cells = map(deskew, cc) # "raddrizza" l'immagine, il testo, le cifre
    cells = preprocess_hog(cells)

    retval, results, neigh_resp, dists = model.find_nearest(cells, 4)

    nn = results.ravel()
    print(nn)
    print(len(nn))

    for r in range(len(cc_ii)):
        for c in range(len(cc_ii[r])):
            if cc_ii[r][c] is not None:
                print("%d %d -> %d"%(r,c,int(nn[cc_ii[r][c]])))

    return nn

def main(image_src, only_m1, show_all, show=False):
    image = cv2.imread(image_src, 0)


    thresh, grid, rr, cc, kernelTransform, cnt_max = method1(image, show_all, show)
    # if not only_m1:
    points, pointsImage = method2(image, cnt_max, kernelTransform, show_all, show)

    # intrecciare dati

    cells = extractCells(points, thresh, kernelTransform)

    nn = extractNumbers(cells)

    # cv2.imshow('source', image)
    # cv2.imshow('gridMask', gridMask)
    # cv2.imshow('grid', grid)

    while 1:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            import os
            folder_name = image_src.split('/')[-1].split('.')[0]
            folder_path = "../esempi/"+folder_name
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            cv2.imwrite(folder_path+'/grid.png',grid)
            if not only_m1:
                cv2.imwrite(folder_path+'/points.png',pointsImage)

                for r in range(len(cells)):
                    for c in range(len(cells[0])):
                        cell = cells[r][c]
                        if cell is None: continue
                        cv2.imwrite(folder_path+'/%d_%d.png'%(r,c),cell)
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
