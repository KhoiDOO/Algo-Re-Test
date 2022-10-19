import os
import numpy as np
import cv2
from kirsch import kirsch_dectect


class scdts:
    def __init__(self, img_state = "vertical", distance = 9, sigmaColor = 75, sigmaSpace = 75):
        self.state = img_state
        self.distance = distance
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
    
    def bilateralFilter(self, img):
        return cv2.bilateralFilter(img, self.distance , self.sigmaColor , self.sigmaSpace)
    
    def img_separating(self, img):
        w, h = img.shape

        return (img[:w//2, :], img[w//2:, :])

    def extract(self, img, path = None):
        rotated_img = None

        if self.state == "vertical":
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img_gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        
        filter_img = self.bilateralFilter(rotated_img)

        bound_img = kirsch_dectect(img=filter_img, filters = (0, 4))

        w, h = bound_img.shape

        upper_bound, lower_bound = self.img_separating(bound_img)

        upper_img, lower_img = self.img_separating(img_gray)

        for x in range(h):
            upper_base = upper_bound[:, x]
            lower_base = lower_bound[:, x]

            max_upper = np.amax(upper_base)
            max_lower = np.amax(lower_base)

            upper_result = np.where(upper_base == max_upper)
            lower_result = np.where(lower_base == max_lower)

            _min = np.amin(lower_result)
            _max = np.amax(upper_result)

            upper_img[:_max, x] = 0 
            # upper_img[_max+1:, x] = 0
            # lower_img[:_min, x] = 0
            lower_img[_min+1:, x] = 0
        
        combined_img = np.concatenate((upper_img, lower_img), axis = 0)

        # cv2.imshow("Results", combined_img)
        # cv2.waitKey(0)

        if path:
            cv2.imwrite(path, combined_img)

# os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path1 = main_data_dir + "\\016_01.bmp"
img_path2 = main_data_dir + "\\016_02.bmp"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

_scdts = scdts(distance = 9, sigmaColor=75, sigmaSpace=75)

_scdts.extract(img1, path = "FingerRegionExtraction\\{}".format(img_path1.split("\\")[-1]))
_scdts.extract(img2, path = "FingerRegionExtraction\\{}".format(img_path2.split("\\")[-1]))
    