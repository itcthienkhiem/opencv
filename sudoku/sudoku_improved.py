import cv2
import numpy as np
from numpy.linalg import norm
import math, logging

logging.basicConfig(filename='sudoku_log', level=logging.DEBUG)

''' assumiamo per semplicita che la griglia di gioco sia l'oggetto con area maggiore '''
def findGrid(thresh):
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    dilate = cv2.dilate(thresh,kernel)

    height, width = dilate.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)

    max_area = -1
    # trovo l'oggetto con area maggiore
    for i in range(max(height, width)):
        r = i%height
        c = i%width
        # diagonale discendente
        if dilate[r, c] == 255:
            area, rect = cv2.floodFill(dilate, mask, (c,r), 64)
            if area > max_area:
                max_area = area
                max_area_point = (c,r)

        c = width-1-c
        # diagonale ascendente
        if dilate[r, c] == 255:
            area, rect = cv2.floodFill(dilate, mask, (c,r), 64)
            if area > max_area:
                max_area = area
                max_area_point = (c,r)

    mask = np.zeros((height+2, width+2), np.uint8)
    area, rect = cv2.floodFill(dilate, mask, max_area_point, 255)

    for r in range(height):
        for c in range(width):
            if dilate[r, c] == 64:
                cv2.floodFill(dilate, mask, (c,r), 0)
            if dilate[r, c] == 255:
                if max_area_point[0] == c and max_area_point[1] == r:
                    continue
                cv2.floodFill(dilate, mask, (c,r), 0)

    grid = cv2.erode(dilate, kernel)

    contours, hierarchy = cv2.findContours(grid.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt_max = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt_max) # (centrox, centroy), (w,h), angolo
    box = cv2.cv.BoxPoints(rect) # ottengo vertici rettangolo
    kernelTransform = np.int0(box)

    height, width = thresh.shape[:2]
    gridMask = np.zeros((height, width), np.uint8)
    cv2.drawContours(gridMask, cnt_max, -1, 255, thickness=cv2.cv.CV_FILLED)
    mask = np.zeros((height+2, width+2), np.uint8)

    tl, tr, br, bl = order_points(kernelTransform)
    cv2.floodFill(gridMask, mask, (int(br[0]+tl[0])/2, int(br[1]+tl[1])/2), 255)
    threshROI = cv2.bitwise_and(thresh, gridMask)

    return threshROI, kernelTransform, cnt_max

def adaptGrid(grid, kernelTransform, cnt_max):
    return four_point_transform(grid, kernelTransform)

    # !!!!!!!!!!!
    height, width = grid.shape[:2]
    for point in kernelTransform:
        if  point[0]<0 or point[0]>=width or \
            point[1]<0 or point[1]>=height:
            # la griglia va fuori dall'immagine
            return four_point_transform(grid, kernelTransform)

    # la griglia sta tutta nell'immagine
    pp = map(lambda p: p[0], cnt_max)
    kernelTransform = order_points(np.int0(pp))
    return four_point_transform(grid, kernelTransform)

def getRows(gridTransformed):
    expandeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    gridClosed = cv2.morphologyEx(gridTransformed, cv2.MORPH_CLOSE, expandeKernel) # piu iterazioni se immagine distorta
    # horizontal = cv2.dilate(gridTransformed, expandeKernel) # espando un po' le linee orizzontali
    # horizontal = cv2.erode(gridTransformed, expandeKernel)

    # horizontalsize = 20 # gridTransformed.shape[1] / 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    horizontal = cv2.erode(gridClosed, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    if show_all: cv2.imshow("horizontal", horizontal)

    cntHorizLines, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rows = list()
    for cnt in cntHorizLines:
        rect = cv2.minAreaRect(cnt) # (centrox, centroy), (w,h), angolo
        box = np.int0(cv2.cv.BoxPoints(rect)) # ottengo vertici rettangolo
        tl, tr, br, bl = order_points(box)
        if (abs(tr[0]-tl[0]) > horizontal.shape[1]//4):
            rows.append({"points":(tl, tr, br, bl), "cnt":cnt})

    rows = map( lambda r: r["cnt"],
                sorted(rows, key=lambda r: r["points"][0][1]))
    return rows

def getCols(gridTransformed):
    expandeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    gridClose = cv2.morphologyEx(gridTransformed, cv2.MORPH_CLOSE, expandeKernel)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    vertical = cv2.erode(gridClose, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    if show_all: cv2.imshow('vertical', vertical)


    cntVertLines, hierarchy = cv2.findContours(vertical.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    verticalLines = list()
    for cnt in cntVertLines:
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
                sorted(cols, key=lambda c: c[0][0]))

    return cols

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

def findPoints(image):
    blur = cv2.GaussianBlur(image,(9,9),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,2)
    cv2.imshow('threshold', thresh)

    grid, kernelTransform, cnt_max = findGrid(thresh)

    gridTransformed = adaptGrid(grid, kernelTransform, cnt_max)
    if show_all: cv2.imshow('gridTransformed', gridTransformed)

    rows = getRows(gridTransformed)
    cols = getCols(gridTransformed)

    height, width = gridTransformed.shape[:2]
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
    return points, pp_canvas, kernelTransform


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


def extractCells(points, image, kernelTransform):
    blur = cv2.GaussianBlur(image,(11,11),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,2)
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

                cv2.imshow("%d,%d"%(r,c), imgCell)

    return gridImages

''' "raddrizza" l'immagine, il testo, le cifre '''
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

''' elimina bordo esterno, riduce la cella ad una dimensione standard (20), elimina
celle vuote ciclando su diagonali e verificando minimo numero di pixel pieni '''
def filterCells(cells):
    cc = list()
    cc_ii = [[None for __ in range(len(cells[0]))] for _ in range(len(cells))]

    ext_dim = 4 #quanti px togliere ai bordi
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            if cell is None: continue
            cell = cell[ext_dim:len(cell)-ext_dim, ext_dim:len(cell[0])-ext_dim]
            cell = cv2.resize(cell, (20,20))

            px_found = 0
            for i in range(20):
                if cell[i][i]:
                    px_found += 1
                if cell[19-i][i]:
                    px_found += 1
                if px_found > 4:
                    break

            if px_found > 4:
                # cv2.imshow("%d %d"%(r,c), cell)
                cc_ii[r][c] = r*9 + c # !!!!!!!!!
                cc.append(cell)

    print("numero celle piene %d"%len(cc))
    return cc, cc_ii


def extractNumbers(cells):
    digits_img = cv2.imread("images/digits.png", 0)
    h, w = digits_img.shape[:2]
    sx, sy = (20, 20) # cell dimensions
    digits = [np.hsplit(row, w//sx) for row in np.vsplit(digits_img, h//sy)]
    digits = np.array(digits)
    digits = digits.reshape(-1, sy, sx) # raggruppo righe

    labels = np.repeat(np.arange(10), len(digits)/10)

    digits2 = map(deskew, digits) # "raddrizza" l'immagine, il testo, le cifre
    samples = preprocess_hog(digits2)

    modelsvm = cv2.SVM()
    params = dict( kernel_type = cv2.SVM_RBF,
                        svm_type = cv2.SVM_C_SVC,
                        C = 2.67,
                        gamma = 5.383 )
    modelsvm.train(samples, labels, params = params)

    modelkn = cv2.KNearest()
    modelkn.train(samples, labels)

    cc, cc_ii = filterCells(cells)

    cells = map(deskew, cc) # "raddrizza" l'immagine, il testo, le cifre
    cells = preprocess_hog(cells)

    #cells = np.float32(cc)
    nn_svm = np.float32( [modelsvm.predict(s) for s in cells])

    retval, results, neigh_resp, dists = modelkn.find_nearest(cells, 4)
    nn_kn = results.ravel()

    return nn_svm, nn_kn

def main(image_src, show_all):
    image = cv2.imread(image_src, 0)

    points, pointsImage, kernelTransform = findPoints(image)

    cells = extractCells(points, image, kernelTransform)
    nn_svm, nn_kn = extractNumbers(cells)

    print("Numeri stimati dal metodo SVM:")
    print(nn_svm)

    print("Numeri stimati dal metodo KN:")
    print(nn_kn)

    while 1:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            import os
            folder_name = image_src.split('/')[-1].split('.')[0]
            folder_path = "output/"+folder_name
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            cv2.imwrite(folder_path+'/grid.png',grid)
            cv2.imwrite(folder_path+'/points.png',pointsImage)

            cells_path = folder_path+'/cells'
            os.mkdir(cells_path)
            for r in range(len(cells)):
                for c in range(len(cells[0])):
                    cell = cells[r][c]
                    if cell is None: continue
                    cv2.imwrite(cells_path+'/%d_%d.png'%(r,c),cell)
            break
        elif k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='OpenCV. Recognise a grid and extract data cells.')
    parser.add_argument('img', metavar='img', nargs='?', default="../images/sudoku.jpg", help='image path')
    parser.add_argument('-a', '--show_all', dest="show_all", nargs='?', const=True, default=False, help='show all result images')
    args = parser.parse_args()

    global show_all
    show_all = args.show_all

    main("../images/%s"%args.img, args.show_all)
