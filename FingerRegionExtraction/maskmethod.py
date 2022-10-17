import os
import numpy as np
import cv2

class MaskVein:
    def __init__(self, mask = None, img_state = 'vertical', img_shape = None) -> None:
        self.state = img_state
        self.shape = img_shape
        if mask:
            self.mask = mask
        else:
            if self.state == 'vertical':
                w, h = self.shape
                h1 = (0, int(h/4))
                h2 = (h1[1] + 1, h1[1]*3)
                h3 = (h2[1] + 1, h - h1[1] + h2[1])
                self.mask = np.zeros((w, h), dtype=np.int8)
                self.mask[:, h1[0]:h1[1]] = -1
                self.mask[:, h2[0]:h2[1]] = 1
                self.mask[:, h3[0]:h3[1]] = -1
                
    def show_mask(self):
        cv2.imshow("Mask", self.mask)
        cv2.waitKey(0)
    
    def get_mask(self):
        return self.mask
    
    def get_roi(self, img : np.ndarray):
        return img * self.mask
    
    def extract(self, img : np.ndarray, gray = "grayscale", path = None):
        if gray == "grayscale":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif gray == "avg":
            img_gray = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
        else:
            img_gray = img[:, :, gray]

        roi = self.get_roi(img_gray)
        # cv2.imshow("ROI", roi)
        # cv2.waitKey(0)
<<<<<<< Updated upstream
        print(np.unique(roi))
=======
        print(np.unique)
>>>>>>> Stashed changes


os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path = main_data_dir + "\\016_01.bmp"
img = cv2.imread(img_path)

maskvein = MaskVein(img_shape=(img.shape[0], img.shape[1]))
maskvein.extract(img)