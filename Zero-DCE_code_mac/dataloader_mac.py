# -*- coding: utf-8 -*-
# @Time    : 2026/3/7 17:38
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : dataloader_mac.py
import os
import glob
import random

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

random.seed(1143)


def populate_train_list(lowlight_images_path):
    image_list_lowlight = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_list_lowlight.extend(glob.glob(os.path.join(lowlight_images_path, ext)))

    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list


class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 256
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = Image.open(data_lowlight_path).convert("RGB")
        # 新版 Pillow 更稳的写法
        data_lowlight = data_lowlight.resize((self.size, self.size), Image.Resampling.LANCZOS)

        data_lowlight = np.asarray(data_lowlight, dtype=np.float32) / 255.0
        data_lowlight = torch.from_numpy(data_lowlight).float()

        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)