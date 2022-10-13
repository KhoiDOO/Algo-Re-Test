import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class BDIP:
    def __init__(self, patch_size = 2, channel_merge = False, epsilon = 0.000001, n_Octaves = 3, threshold = 0.1) -> None:
        self.block_size = patch_size
        self.channel_merge = channel_merge
        self.patch_area = patch_size**2
        self.epsilon = epsilon

    def padding(self, img : np.array):
        padding_size = [0, 0]
        if img.shape[0] % self.block_size != 0:
            padding_size[0] = self.block_size - img.shape[0] % self.block_size
        elif img.shape[1] % self.block_size != 0:
            padding_size[1] = self.block_size - img.shape[1] % self.block_size
        return padding_size

    def add_padding(self, gray : np.array, padding_size : tuple):
        row_padding = np.zeros((gray.shape[0], padding_size[1])) 
        col_padding = np.zeros((padding_size[0], gray.shape[1] + padding_size[1]))
        row_pad_gray = np.concatenate([gray, row_padding], axis=1)
        full_padd_gray = np.concatenate([row_pad_gray, col_padding], axis = 0)
        return full_padd_gray


    def extract(self, img : np.array, gray = "grayscale", path = None):
        if gray == "grayscale":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif gray == "avg":
            img_gray = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
        else:
            img_gray = img[:, :, gray]

        padding_size = self.padding(img_gray)
        padd_gray = self.add_padding(img_gray, padding_size)

        row_index, col_index = padd_gray.shape[0], padd_gray.shape[1]
        row_block_index, col_block_index = int(row_index/self.block_size), int(col_index/self.block_size)

        output = np.zeros((row_block_index, col_block_index))

        for i in range(0, row_index, self.block_size):
            for j in range(0, col_index, self.block_size):
                current_index = (int(i/self.block_size), int(j/self.block_size))
                current_block = padd_gray[i:i+self.block_size, j:j+self.block_size]
                output[current_index[0], current_index[1]] = self.patch_area - np.sum(current_block)/(np.amax(current_block) + self.epsilon)

        # cv2.imshow("results", output)
        # cv2.waitKey(0)
        if path:
            frame_normed = 255 * (output - output.min()) / (output.max() - output.min())
            frame_normed = np.array(frame_normed, np.int)
            cv2.imwrite(path, frame_normed)
            plt.hist(output.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k') #calculating histogram
            plt.savefig("Histogram\\" + path.split("\\")[-1])
            


os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path1 = main_data_dir + "\\CHGastro_Abnormal_037.png"
img_path2 = main_data_dir + "\\CHGastro_Normal_047.png"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

bdip = BDIP()
bdip.extract(img = img1, path="ExampleImage\\BDIP_" + img_path1.split("\\")[-1])                  
bdip.extract(img = img2, path="ExampleImage\\BDIP_" + img_path2.split("\\")[-1])

bdip_img1_path = main_data_dir + "\\BDIP_CHGastro_Abnormal_037"