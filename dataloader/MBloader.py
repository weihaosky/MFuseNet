import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import preprocess 

class myImageFloder(data.Dataset):
    def __init__(self, bins, img, disp, mode, dispRange, training, augment=False):
        self.mode = mode
        self.left = bins[0]
        self.right = bins[1]
        if mode == '5view':
            self.bottom = bins[2]
            self.top = bins[3]
        self.img = img
        self.disp = disp
        self.training = training
        self.augment = augment
        self.dispRange = dispRange

    def __getitem__(self, index):
        left_path  = self.left[index]
        right_path = self.right[index]
        if self.mode == '5view':
            bottom_path = self.bottom[index]
            top_path = self.top[index]
        img_path = self.img[index]
        disp_path = self.disp[index]
        d = self.dispRange

        img = Image.open(img_path)
        w, h = img.size
        img = np.array(img).transpose(2, 0, 1)

        disp = Image.open(disp_path)
        disp = np.array(disp)

        left_mem = np.memmap(left_path, dtype=np.float32, shape=(1, d, h, w))
        right_mem = np.memmap(right_path, dtype=np.float32, shape=(1, d, h, w))

        left = np.squeeze(np.array(left_mem))
        right = np.squeeze(np.array(right_mem))
        left[np.isnan(left)]=20
        right[np.isnan(right)]=20

        if self.mode == '5view':
            bottom = np.memmap(bottom_path, dtype=np.float32, shape=(1, d, w, h))
            top = np.memmap(top_path, dtype=np.float32, shape=(1, d, w, h))
            bottom=np.rot90(np.array(bottom), k=-1, axes=(2,3)).copy()
            top=np.rot90(np.array(top), k=-1, axes=(2,3)).copy()
            bottom[np.isnan(bottom)]=20
            top[np.isnan(top)]=20
            bottom = np.squeeze(bottom)
            top = np.squeeze(top)

        if self.training:  
            th, tw = 256, 256

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left = left[:, y1:y1+th, x1:x1+tw]
            right = right[:, y1:y1+th, x1:x1+tw]
            if self.mode == '5view':
                bottom = bottom[:, y1:y1+th, x1:x1+tw]
                top = top[:, y1:y1+th, x1:x1+tw]
            img = img[:, y1:y1+th, x1:x1+tw]
            disp = disp[y1:y1+th, x1:x1+tw] 
            disp = np.float32(disp / 4.0) 

            if self.mode == '3view':
                return left, right, img, disp
            elif self.mode == '5view':
                return left, right, bottom, top, img, disp

        else:
            if self.mode == '3view':
                return left, right
            elif self.mode == '5view':
                return left, right, bottom, top


    def __len__(self):
        return len(self.left)
