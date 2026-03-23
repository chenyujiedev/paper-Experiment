# -*- coding: utf-8 -*-
# @Time    : 2026/3/20 11:08
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : h_test.py
# -*- coding: utf-8 -*-
# @Time    : 2026/3/7 15:20
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : lowlight_test_mac.py

import torch
import torchvision
import os
import model_mac
import numpy as np
from PIL import Image
import glob
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# 只加载一次模型
DCE_net = model_mac.enhance_net_nopool().to(device)
DCE_net.load_state_dict(torch.load('snapshots/Epoch8.pth', map_location=device))
DCE_net.eval()


def lowlight(image_path, save_root='result_8'):
    data_lowlight = Image.open(image_path).convert('RGB')
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)
        enhanced_image = torch.clamp(enhanced_image, 0, 1)
    end_time = time.time() - start
    print("Inference time:", end_time)

    filename = os.path.basename(image_path)
    result_path = os.path.join(save_root, filename)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    torchvision.utils.save_image(enhanced_image.cpu(), result_path)


if __name__ == '__main__':

    filePath = 'data/train_data'
    test_list = glob.glob(os.path.join(filePath, '*.png')) + \
                glob.glob(os.path.join(filePath, '*.jpg')) + \
                glob.glob(os.path.join(filePath, '*.jpeg'))

    for image in sorted(test_list):
        print(image)
        lowlight(image)