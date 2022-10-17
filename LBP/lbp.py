import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class LBP:
    def __init__(self, binary_direction = "clockwise", block_size = (3, 3)):
        self.direction = binary_direction
        self.block_size = block_size
    
    def get_pixel(self, img, center, x, y):
        new_value = 0
        try:
            # If local neighbourhood pixel 
            # value is greater than or equal
            # to center pixel values then 
            # set it to 1

            if img[x][y] >= center:
                new_value = 1
        except:
            # Exception is required when 
            # neighbourhood value of a center
            # pixel value is null i.e. values
            # present at boundaries.
            pass

        return new_value
    
    def lbp_calculated_pixel(self, img, x, y):
   
        center = img[x][y]
   
        val_ar = []
      
        val_ar.append(self.get_pixel(img, center, x-1, y-1))
      
        val_ar.append(self.get_pixel(img, center, x-1, y))
      
        val_ar.append(self.get_pixel(img, center, x-1, y + 1))
      
        val_ar.append(self.get_pixel(img, center, x, y + 1))
      
        val_ar.append(self.get_pixel(img, center, x + 1, y + 1))
      
        val_ar.append(self.get_pixel(img, center, x + 1, y))
      
        val_ar.append(self.get_pixel(img, center, x + 1, y-1))
      
        val_ar.append(self.get_pixel(img, center, x, y-1))
       
        # values to decimal
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
   
        val = 0
      
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
          
        return val
    
    def extract (self, img : np.array, gray = "grayscale", path = None):
        if gray == "grayscale":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif gray == "avg":
            img_gray = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
        else:
            img_gray = img[:, :, gray]
        
        height, width = img_gray.shape

        img_lbp = np.zeros((height, width), np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = self.lbp_calculated_pixel(img_gray, i, j)
        
        if path:
            cv2.imwrite(path, img_lbp)
    
os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path1 = main_data_dir + "\\016_01.bmp"
img_path2 = main_data_dir + "\\016_01.bmp"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

lbp = LBP()
lbp.extract(img = img1, path="ExampleImage\\LBP_vein_" + img_path1.split("\\")[-1])                  
lbp.extract(img = img2, path="ExampleImage\\LBP_vein" + img_path2.split("\\")[-1])