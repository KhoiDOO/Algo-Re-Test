import os
import numpy as np
import cv2

os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path1 = main_data_dir + "\\016_01.bmp"
img_path2 = main_data_dir + "\\016_02.bmp"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

blur1 = cv2.bilateralFilter(img1,9,75,75)
blur2 = cv2.bilateralFilter(img2,9,75,75)

print(blur1.shape, blur2.shape)

cv2.imshow("Blur 1", blur1)
cv2.imshow("Blur 2", blur2)

cv2.imwrite("ImageSmoothing/{}".format(img_path1.split("\\")[-1]), blur1)
cv2.imwrite("ImageSmoothing/{}".format(img_path2.split("\\")[-1]), blur2)