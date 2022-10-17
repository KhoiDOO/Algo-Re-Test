from skimage.feature import graycomatrix, graycoprops
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SlowGLCM:
    def __init__(self, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True) -> None:
        self.dists = dists
        self.agls = agls
        self.lvl = lvl
        self.sym = sym
        self.norm = norm
    
    def make_glcm(self, img: np.array, gray = "grayscale", _return = True):
        if gray == "grayscale":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif gray == "avg":
            img_gray = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
        else:
            img_gray = img[:, :, gray]

        glcm = graycomatrix(img_gray, 
                        distances=self.dists, 
                        angles=self.agls, 
                        levels=self.lvl,
                        symmetric=self.sym, 
                        normed=self.norm)
        
        self.glcm = glcm
        if _return:
            return self.glcm
    
    def glcm_coprops(self, img, gray = "grayscale", prop = 'contrast' ,_return = True):
        glcm = self.make_glcm(img = img, gray = gray, _return = True)

        coprops = graycoprops(glcm, prop)
        if _return:
            return coprops


    

os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path = main_data_dir + "\\CHGastro_Abnormal_037.png"
img = cv2.imread(img_path)

glcm = SlowGLCM()
# print(glcm.make_glcm(img))
print(glcm.glcm_coprops(img))

