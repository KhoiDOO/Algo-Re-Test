import numpy as np
import cv2

class BDIP:
    def __init__(self, patch_size, channel_merge = False, n_Octaves = 3) -> None:
        self.block_size = patch_size
        self.channel_merge = channel_merge
    
    def extract(self, img : np.array, gray = "grayscale"):
        if gray == "grayscale":
            img = 


            
        