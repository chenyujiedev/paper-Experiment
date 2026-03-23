# -*- coding: utf-8 -*-
# @Time    : 2026/3/7 15:20
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : lowlight_test_mac.py
# Zero-DCE: 推理复现
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


def lowlight(image_path):
    data_lowlight = Image.open(image_path).convert('RGB')

    data_lowlight = np.asarray(data_lowlight) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0).to(device)

    DCE_net = model_mac.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load('snapshots_darkexp_highprotect/Epoch2.pth', map_location=device))
    DCE_net.eval()

    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start
    print("Inference time:", end_time)

    image_path = image_path.replace('data', '3_22/light_loss2.0')
    result_path = image_path

    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torchvision.utils.save_image(enhanced_image.cpu(), result_path)


if __name__ == '__main__':
    with torch.no_grad():
        filePath = 'data/LOLv1/Test/input'
        # filePath = 'data/train_data'
        test_list = glob.glob(os.path.join(filePath, '*.png')) + \
                    glob.glob(os.path.join(filePath, '*.jpg')) + \
                    glob.glob(os.path.join(filePath, '*.jpeg'))

        for image in sorted(test_list):
            print(image)
            lowlight(image)
