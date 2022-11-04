import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class BVLC:
    def __init__(self, patchsize = 2, channel_merge = False, stride = 1, n_Octaves = 3, epsilon = 0.000001, pairs = ((0, 1), (1, 0), (1, 1), (1, -1))) -> None:
        """__init__ _summary_

        Keyword Arguments:
            channel_merge -- boolean: all channel in the image will be merged if True (default: {False})
            stride -- int: the translating size of block size  (default: {1})
            n_Octaves -- int: the number of times the kernel scales - not implemented (default: {3})
            epsilon -- float: relatively small number to avoid zero division (default: {0.000001})
            pairs -- tuple (4 x 2): Direction of block translating (default: {((0, 1), (1, 0), (1, 1), (1, -1))})
            patchsize -- int: size of the block
        """        
        self.block_size = patchsize
        self.patch_area = self.block_size ** 2
        self.channel_merge = channel_merge
        self.pairs = pairs
        self.stride = stride
        self.epsilon = epsilon

    def padding(self, img : np.array):
        """padding Calculates the padding size

        Arguments:
            img -- np.array: input image

        Returns:
            padding size -- tuple(4 x 1): padding size for each side of input image
        """
        padding_size = [0, 0, 0, 0] # up down left right
        if img.shape[0] % self.block_size == 0:
            padding_size[0], padding_size[1] = self.block_size, self.block_size
        elif img.shape[0] % self.block_size == 1:
            padding_size[0] = self.block_size
        if img.shape[1] % self.block_size == 0:
            padding_size[2], padding_size[3] = self.block_size, self.block_size
        return padding_size

    def add_padding(self, gray : np.array, padding_size : tuple):
        """add_padding add padding to the image

        Arguments:
            gray -- np.array: gray scale input image
            padding_size -- tuple(4 x 1): padding size need adding 

        Returns:
            padded image
        """
        up_pad = np.zeros((padding_size[0], gray.shape[1]))
        down_pad = np.zeros((padding_size[1], gray.shape[1]))
        left_pad = np.zeros((gray.shape[0] + padding_size[0] + padding_size[1], padding_size[2]))
        right_pad = np.zeros((gray.shape[0] + padding_size[0] + padding_size[1], padding_size[3]))  
        up_cat_pad = np.concatenate((up_pad, gray), axis = 0)
        down_cat_pad = np.concatenate((up_cat_pad, down_pad), axis = 0)
        left_cat_pad = np.concatenate((left_pad, down_cat_pad), axis = 1)
        right_cat_pad = np.concatenate((left_cat_pad, right_pad), axis = 1)
        return right_cat_pad

    def local_coree_coef(self, current_block : np.array, shift_block : np.array):
        """local_coree_coef Calculates the local correlation coefficient

        Arguments:
            current_block -- np.array(block_size x block_size): sub-matrix of image overlap the current block
            shift_block -- np.array(block_size x block_size): sub-matrix of image translated by pair

        Returns:
            local_coree_coef -- float
        """
        local_mean = np.mean(current_block)
        local_std = np.std(current_block)
        shift_mean = np.mean(shift_block)
        shift_std = np.std(shift_block)      

        nominator = np.sum(current_block*shift_block)/self.patch_area - local_mean*shift_mean
        dominator = local_std * shift_std + self.epsilon

        return nominator/dominator

    def extract(self, img : np.array, gray = "grayscale", path = None, extended = True):
        """extract Calculates the BVLC image

        Arguments:
            img -- np.array: input image

        Keyword Arguments:
            gray -- str: define the way of calculating the grayscale image (default: {"grayscale"})
            path -- str: path to save the BVLC image and its histogram (default: {None})
        """
        if gray == "grayscale":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif gray == "avg":
            img_gray = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
        else:
            img_gray = img[:, :, gray]

        padding_size = self.padding(img_gray)

        padd_gray = self.add_padding(img_gray, padding_size)

        row_index, col_index = padd_gray.shape[0], padd_gray.shape[1]

        row_block_index, col_block_index = int((row_index - 1)/self.block_size), int((col_index - 1)/self.block_size)

        output = np.zeros((row_block_index, col_block_index))

        for i in range(self.block_size, row_index - self.block_size, self.block_size):
            for j in range(self.block_size, col_index - self.block_size, self.block_size):
                current_index = (int(i/self.block_size), int(j/self.block_size))
                current_block = padd_gray[i:i+self.block_size, j:j+self.block_size]
                local_coefs = []
                for x in self.pairs:
                    i_shift = i + x[1]
                    j_shift = j + x[0]
                    shift_block = padd_gray[i_shift : i_shift + self.block_size, j_shift : j_shift + self.block_size]
                    local_coefs.append(self.local_coree_coef(current_block, shift_block))
                output[current_index[0], current_index[1]] = max(local_coefs) - min(local_coefs)
        # cv2.imshow("results", output)
        # cv2.waitKey(0)
        if path:
            frame_normed = 255 * (output - output.min()) / (output.max() - output.min())
            frame_normed = np.array(frame_normed, np.int)
            if extended:
                base_img = np.zeros((frame_normed.shape[0], frame_normed.shape[1], 3))
                base_img[:, :, gray] = frame_normed
                frame_normed = base_img
            cv2.imwrite(path, frame_normed)
            plt.hist(output.ravel(), bins=256, range=(0.0, 2.0), fc='k', ec='k') #calculating histogram
            plt.savefig("Histogram\\" + path.split("\\")[-1])


os.chdir("..")
main_data_dir = os.getcwd() + "\\ExampleImage"
img_path1 = main_data_dir + "\\CHGastro_Abnormal_037.png"
img_path2 = main_data_dir + "\\CHGastro_Normal_047.png"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

bdlc = BVLC(patchsize=7)
extract1 = bdlc.extract(img = img1, gray = 2, path="ExampleImage\\BVLC_7_red" + img_path1.split("\\")[-1])                  
extract2 = bdlc.extract(img = img2, gray = 2, path="ExampleImage\\BVLC_7_red" + img_path2.split("\\")[-1])