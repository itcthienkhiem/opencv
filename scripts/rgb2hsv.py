
NORMAL_CONVERSION = 0
OPENCV_CONVERSION = 1

def convert(colorRGB, flag=NORMAL_CONVERSION):
    if len(colorRGB)!=3:
        raise Exception()
    r = colorRGB[0]
    g = colorRGB[1]
    b = colorRGB[2]

    if flag == OPENCV_CONVERSION:
        # Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
        import cv2
        import numpy as np
        color = np.uint8([[ [b,g,r] ]])
        hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
        return hsv_color[0][0]

    r1 = float(r)/255
    g1 = float(g)/255
    b1 = float(b)/255

    cmax = max(r1,g1,b1)
    cmin = min(r1,g1,b1)

    delta = cmax - cmin

    if delta == 0:
        h = 0
    elif cmax == r1:
        h = 60*(( (g1 - b1)/delta )%6)
    elif cmax == g1:
        h = 60*(( (b1 - r1)/delta )+2)
    elif cmax == b1:
        h = 60*(( (r1 - g1)/delta )+4)
    else:
        raise Exception()

    if cmax == 0:
        s = 0
    else:
        s = delta/cmax

    v = cmax

    return [h,s*100,v*100]



if __name__ == '__main__':
    colorRGB = [0,200,0]
    print(convert(colorRGB))
    print(convert(colorRGB, OPENCV_CONVERSION))
