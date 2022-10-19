import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


m1 = np.array([[5, 5, 5], 
                [-3, 0, -3], 
                [-3, -3, -3]])
m2 = np.array([[-3, 5, 5], 
                [-3, 0, 5], 
                [-3,-3,-3]])
m3 = np.array([[-3, -3, 5], 
                [-3, 0, 5], 
                [-3, -3, 5]])
m4 = np.array([[-3,-3,-3], 
                [-3, 0, 5], 
                [-3, 5, 5]])
m5 = np.array([[-3, -3, -3], 
                [-3, 0, -3], 
                [5, 5, 5]])
m6 = np.array([[-3, -3, -3], 
                [5, 0, -3], 
                [5, 5, -3]])
m7 = np.array([[5, -3, -3], 
                [5, 0, -3], 
                [5, -3, -3]])
m8 = np.array([[5, 5, -3], 
                [5, 0, -3], 
                [-3, -3, -3]])

filterlist = [m1, m2, m3, m4, m5, m6, m7, m8]

def kirsch_dectect(img, filters = (2, 6), range = 'separated'):

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernels = []

    if range == 'separated':
        kernels = [filterlist[filters[0]], filterlist[filters[1]]]
    elif range == 'sequence':
        kernels = filterlist[filters[0] : filters[1] + 1]
    
    filtered_list = np.zeros((8, img_gray.shape[0], img_gray.shape[1]))

    # print(filtered_list.shape)

    for k in kernels:
        out = cv.filter2D(src = img_gray, ddepth = -1, kernel = k)
        filtered_list[k] = out
    
    final = np.max(filtered_list, axis = 0)
    
    final[ np.where(final >= 255) ] = 255

    final[ np.where(final < 255) ] = 0

    return final

os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path1 = main_data_dir + "\\016_01.bmp"
img_path2 = main_data_dir + "\\016_02.bmp"
img1 = cv.imread(img_path1)
img2 = cv.imread(img_path2)

# img1 = cv.rotate(img1, cv.ROTATE_90_CLOCKWISE)
# img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)

edge1 = kirsch_dectect(img1)
edge2 = kirsch_dectect(img2)

cv.imwrite("EdgeDetection/verkernel_{}".format(img_path1.split("\\")[-1]), edge1)
cv.imwrite("EdgeDetection/verkernel_{}".format(img_path2.split("\\")[-1]), edge2)
