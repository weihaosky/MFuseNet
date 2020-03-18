import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

# Disp_type = ''    # the ground-truth disparity format is disparity*1
Disp_type = 'x4'    # the ground-truth disparity format is disparity*4

def dataloader(filepath, mode='3view'):
    left_train = []
    right_train = []
    bottom_train = []
    top_train = []
    img_train = []
    disp_train = []
    if mode == '3view':
        for scene in os.listdir(filepath):
            left_train.append(filepath + scene + '/left.bin')
            right_train.append(filepath + scene + '/right.bin')
            img_train.append(filepath + scene + '/view1.png')
            disp_train.append(filepath + scene + '/disp1' + Disp_type + '.png')

        return left_train, right_train, img_train, disp_train

    elif mode == '5view':
        for scene in os.listdir(filepath):
            left_train.append(filepath + scene + '/left.bin')
            right_train.append(filepath + scene + '/right.bin')
            bottom_train.append(filepath + scene + '/bottom.bin')
            top_train.append(filepath + scene + '/top.bin')
            img_train.append(filepath + scene + '/view1.png')
            disp_train.append(filepath + scene + '/disp1' + Disp_type + '.png')
                    
        return left_train, right_train, bottom_train, top_train, img_train, disp_train



    