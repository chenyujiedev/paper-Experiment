import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


def _gather_images(img_dir):
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(img_dir, p)))
    files.sort()
    return files


class MemoryFriendlyLoader(data.Dataset):
    def __init__(self, img_dir, task):
        self.task = task
        self.train_low_data_names = _gather_images(img_dir)

        if len(self.train_low_data_names) == 0:
            raise RuntimeError(f'No images found in {img_dir}')

        random.shuffle(self.train_low_data_names)

    def __getitem__(self, index):
        img_path = self.train_low_data_names[index]
        img_name = os.path.basename(img_path)

        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.task == 'test':
            return img, img_name
        else:
            return img

    def __len__(self):
        return len(self.train_low_data_names)