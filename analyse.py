import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import canny
from os import walk

imgs = []
for (dirpath, dirnames, filenames) in walk("input_images/"):
    imgs.extend(filenames)
    break

for pic in imgs:
    img = cv2.imread("input_images/"+pic, 0)

    kernal = np.array(([0,1,0],[1,1,1],[0,1,0]), dtype="uint8")
    edges = canny(img,low_threshold=50,high_threshold=100)


    edges = np.array(edges, dtype=np.uint8)
    opening = cv2.dilate(edges,kernal,iterations = 1)


    opening = 255 - opening

    blr = cv2.blur(opening,(3,3));

    gray = np.float32(blr)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    max = 0
    mxind = (0,0)
    count = 0

    for i in range(0, 512 - 120, 32):
        for j in range(0, 512 - 120, 32):
            count = 0
            for k in range(120):
                for l in range(120):
                    if (dst[i+k][j+l] > 0):
                        count += dst[i+k][j+l]*1.5
                    elif (dst[i+k][j+l] < 0):
                        count += dst[i+k][j+l]*-3
            if (count > max):
                max = count
                mxind = (i,j)


    count = 0
    chng = 0
    for i in range(120):
        if (opening[i + mxind[0]][i + mxind[1]] == 254 and chng == 1):
            chng = 0
            count += 1
        elif (opening[i + mxind[0]][i + mxind[1]] == 255 and chng == 0):
            chng = 1


    max = count

    count = 0
    chng = 0

    for i in range(120):
        if (opening[i + mxind[0]][(120-i) + mxind[1]] == 254 and chng == 1):
            chng = 0
            count += 1
        elif (opening[i + mxind[0]][(120-i) + mxind[1]] == 255 and chng == 0):
            chng = 1
            
    output = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    

    for i in range(120):
        output[mxind[0]][i + mxind[1]][2] = 255
        output[120 + mxind[0]][i + mxind[1]][2] = 255
        output[1 + mxind[0]][i + mxind[1]][2] = 255
        output[121 + mxind[0]][i + mxind[1]][2] = 255
        output[i + mxind[0]][mxind[1]][2] = 255
        output[i + mxind[0]][120 + mxind[1]][2] = 255
        output[i + mxind[0]][1 + mxind[1]][2] = 255
        output[i + mxind[0]][121 + mxind[1]][2] = 255

    if (count > max):
        max = count
        
        if (max % 2 != 0):
            max+=1
        max = max/2
        
        for i in range(120):
            output[i + mxind[0]][(120-i) + mxind[1]][2] = 255
            output[i + mxind[0]][(120-i) + mxind[1]+1][2] = 255
            output[i + mxind[0]][(120-i) + mxind[1]+2][2] = 255
            output[i + mxind[0]][(120-i) + mxind[1]+3][2] = 255
            cv2.putText(img=output, text=str(max), org=(mxind[1], mxind[0]+30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)

    else :
        if (max % 2 != 0):
            max+=1
        max = max/2
        for i in range(120):
            output[i + mxind[0]][i + mxind[1]][2] = 255
            output[i + mxind[0]][i + mxind[1]+1][2] = 255
            output[i + mxind[0]][i + mxind[1]+2][2] = 255
            output[i + mxind[0]][i + mxind[1]+3][2] = 255
            cv2.putText(img=output, text=str(max), org=(mxind[1]+50, mxind[0]+30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)

    

    print(max)

    cv2.imwrite("output_images/"+pic,output)

 
