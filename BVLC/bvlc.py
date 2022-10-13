import os
import numpy as np
import cv2

class BVLC:
    def __init__(self, channel_merge = False, n_Octaves = 3, pairs = ((0, 1), (1, 0), (1, 1), (1, -1))) -> None:
        self.block_size = 2
        self.patch_area = self.block_size ** 2
        self.channel_merge = channel_merge
        self.pairs = pairs

    def padding(self, img : np.array):
        padding_size = [0, 0, 0, 0] # up down left right
        if img.shape[0] % self.block_size == 0:
            padding_size[0], padding_size[1] = 1, 1
        elif img.shape[0] % self.block_size == 1:
            padding_size[0] = 1
        if img.shape[1] % self.block_size == 0:
            padding_size[2], padding_size[3] = 1, 1
        return padding_size

    def add_padding(self, gray : np.array, padding_size : tuple):
        up_pad = np.zeros((padding_size[0], gray.shape[1]))
        print(up_pad.shape)
        down_pad = np.zeros((padding_size[1], gray.shape[1]))
        print(down_pad.shape)
        left_pad = np.zeros((gray.shape[0] + padding_size[0] + padding_size[1], padding_size[2]))
        print(left_pad.shape)
        right_pad = np.zeros((gray.shape[0] + padding_size[0] + padding_size[1], padding_size[3]))  
        print(right_pad.shape)
        up_cat_pad = np.concatenate((up_pad, gray), axis = 1)
        down_cat_pad = np.concatenate((up_cat_pad, down_pad), axis = 1)
        left_cat_pad = np.concatenate((left_pad, down_cat_pad), axis = 0)
        right_cat_pad = np.concatenate((left_cat_pad, right_pad), axis = 0)
        return right_cat_pad

    def local_coree_coef(self, current_block : np.array, shift_block : np.array):
        local_mean = np.mean(current_block)
        local_std = np.std(current_block)
        shift_mean = np.mean(shift_block)
        shift_std = np.std(shift_block)

        # print(local_mean, shift_mean, shift_std, local_std)

        return (np.sum(current_block * shift_block) - local_mean*shift_mean)/(local_std * shift_std * pow(self.patch_area, 2))

    def extract(self, img : np.array, gray = "grayscale"):
        if gray == "grayscale":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif gray == "avg":
            img_gray = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
        else:
            img_gray = img[:, :, gray]

        padding_size = self.padding(img_gray)
        print(padding_size)

        padd_gray = self.add_padding(img_gray, padding_size)

        row_index, col_index = padd_gray.shape[0], padd_gray.shape[1]

        print(row_index, col_index)

        # row_block_index, col_block_index = int((row_index - 1)/self.block_size), int((col_index - 1)/self.block_size)

        # output = np.zeros((row_block_index, col_block_index))

        # for i in range(0, row_index - 1, self.block_size):
        #     for j in range(0, col_index - 1, self.block_size):
        #         current_index = (int(i/self.block_size), int(j/self.block_size))
        #         current_block = padd_gray[i:i+self.block_size, j:j+self.block_size]
        #         local_coefs = []
        #         for x in self.pairs:
        #             i_shift = i + x[1]
        #             j_shift = j + x[0]
        #             print(i_shift, j_shift)
        #             shift_block = padd_gray[i_shift : i_shift + self.block_size, j_shift : j_shift + self.block_size]
        #             print(current_block.shape, shift_block.shape)
        #             # local_coefs.append(self.local_coree_coef(current_block, shift_block))
        #         break
        #         output[current_index[0], current_index[1]] = max(local_coefs) - min(local_coefs)
        #     break

        # cv2.imshow("results", output)
        # cv2.waitKey(0)
        # return output


os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path1 = main_data_dir + "\\CHGastro_Abnormal_037.png"
img_path2 = main_data_dir + "\\CHGastro_Normal_047.png"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

bdlc = BVLC()
extract1 = bdlc.extract(img1) 
# cv2.imwrite("ExampleImage\\BDIP_" + img_path1.split("\\")[-1], extract1)                  
# extract2 = bdlc.extract(img2)
# cv2.imwrite("ExampleImage\\BDIP_" + img_path2.split("\\")[-1], extract2)


