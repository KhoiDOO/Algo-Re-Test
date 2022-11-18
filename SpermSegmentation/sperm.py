import os 
import cv2 as cv
from scipy import ndimage
import numpy as np

print(os.getcwd())

img = ['50frame15.jpg', 'frame2.jpg']

img0 = cv.imread(img[0])
img1 = cv.imread(img[1])
# print(img0.shape)

gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

# cv.imshow('check', img0)
# cv.waitKey(0)

blur = cv.GaussianBlur(gray,(11,11),0)
# cv.imshow('gaussion', blur)
# cv.waitKey(0)

LoG = ndimage.gaussian_laplace(blur, sigma=1)
# print(np.unique(LoG))
# cv.imshow('LoG', LoG)
# cv.waitKey(0)

thresh = cv.threshold(LoG, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
# print(thresh.shape)
# print(np.unique(thresh))
cv.imshow('spot', thresh)
cv.waitKey(0)

# test_arr_1 = np.array([[1, 2], [2, 1]])
# test_arr_2 = np.array([[1, 2], [2, 1]])

# test_arr_out = np.multiply(test_arr_1, test_arr_2)
# print(test_arr_out)

thresh[thresh > 0] = 1

# output_mul = np.zeros(img0.shape)
# # for i in range(3):
# #     output_mul[:, :, i] = np.multiply(thresh, img0[:, :, i])
# # print(output_mul.shape)
# cv.imshow('result', output_mul)
# cv.waitKey(0)