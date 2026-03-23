# -*- coding: utf-8 -*-
# @Time    : 2026/3/16 01:43
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : calc_niqe_brisque_loe_79.py

import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from PIL import Image
import torch
import pyiqa


def collect_image_paths(folder):
    return natsorted(
        glob(os.path.join(folder, '*.png')) +
        glob(os.path.join(folder, '*.jpg')) +
        glob(os.path.join(folder, '*.jpeg')) +
        glob(os.path.join(folder, '*.bmp'))
    )


def load_rgb_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)


def pil_to_tensor(img_np):
    # HWC [0,255] -> BCHW [0,1]
    img = torch.from_numpy(img_np).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img


def local_mean_luminance(img, size=(50, 50)):
    # img: RGB uint8
    img = img.astype(np.float32) / 255.0
    # 常用做法：取每个像素三个通道最大值作为亮度
    lum = np.max(img, axis=2)
    lum_small = cv2.resize(lum, size, interpolation=cv2.INTER_AREA)
    return lum_small


def calculate_loe(original_img, enhanced_img):
    """
    一个论文里常见的 LOE 实现思路：
    比较原图和增强图在局部亮度上的相对顺序变化。
    返回值越小越好。
    """
    x = local_mean_luminance(original_img)
    y = local_mean_luminance(enhanced_img)

    h, w = x.shape
    n = h * w

    x = x.reshape(-1)
    y = y.reshape(-1)

    loe_sum = 0
    for i in range(n):
        rx = x[i] >= x
        ry = y[i] >= y
        loe_sum += np.sum(rx != ry)

    loe_value = loe_sum / n
    return float(loe_value)


def evaluate_model(model_name, pred_dir, input_dir, niqe_metric, brisque_metric):
    pred_paths = collect_image_paths(pred_dir)
    input_paths = collect_image_paths(input_dir)

    if len(pred_paths) == 0:
        raise FileNotFoundError(f'No prediction images found in: {pred_dir}')
    if len(input_paths) == 0:
        raise FileNotFoundError(f'No input images found in: {input_dir}')
    if len(pred_paths) != len(input_paths):
        raise ValueError(
            f'Image count mismatch for {model_name}: pred={len(pred_paths)}, input={len(input_paths)}'
        )

    niqe_list = []
    brisque_list = []
    loe_list = []

    print(f'\nEvaluating {model_name}')
    print(f'Prediction folder: {pred_dir}')
    print(f'Input folder     : {input_dir}')
    print(f'Number of images : {len(pred_paths)}')

    for pred_path, input_path in zip(pred_paths, input_paths):
        pred_name = os.path.basename(pred_path)
        input_name = os.path.basename(input_path)

        pred_img = load_rgb_image(pred_path)
        input_img = load_rgb_image(input_path)

        if pred_img.shape != input_img.shape:
            raise ValueError(
                f'Shape mismatch: {pred_name} {pred_img.shape} vs {input_name} {input_img.shape}'
            )

        pred_tensor = pil_to_tensor(pred_img)

        with torch.no_grad():
            niqe_score = niqe_metric(pred_tensor).item()
            brisque_score = brisque_metric(pred_tensor).item()

        loe_score = calculate_loe(input_img, pred_img)

        niqe_list.append(niqe_score)
        brisque_list.append(brisque_score)
        loe_list.append(loe_score)

        print(
            f'{pred_name:20s}  '
            f'NIQE={niqe_score:.4f}  '
            f'BRISQUE={brisque_score:.4f}  '
            f'LOE={loe_score:.4f}'
        )

    avg_niqe = float(np.mean(niqe_list))
    avg_brisque = float(np.mean(brisque_list))
    avg_loe = float(np.mean(loe_list))

    print('-' * 70)
    print(f'{model_name} Average NIQE    : {avg_niqe:.4f}')
    print(f'{model_name} Average BRISQUE : {avg_brisque:.4f}')
    print(f'{model_name} Average LOE     : {avg_loe:.4f}')
    print('-' * 70)

    return avg_niqe, avg_brisque, avg_loe


if __name__ == '__main__':
    # 原始79张低照度输入图
    input_dir = 'data/real_lowlight_79'

    # 三个模型的输出
    zero_dir = 'data/result_improved_dark8'
    ruas_dir = 'data/result_ruas_79'
    retinex_dir = 'data/result_retinex_79'

    # pyiqa 指标
    device = 'cpu'   # 这些指标直接用 cpu 就够了，省事稳定
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    brisque_metric = pyiqa.create_metric('brisque', device=device)

    zero_niqe, zero_brisque, zero_loe = evaluate_model(
        'Zero-DCE (improved_Epoch 8)', zero_dir, input_dir, niqe_metric, brisque_metric
    )

    ruas_niqe, ruas_brisque, ruas_loe = evaluate_model(
        'RUAS', ruas_dir, input_dir, niqe_metric, brisque_metric
    )

    ret_niqe, ret_brisque, ret_loe = evaluate_model(
        'Retinexformer', retinex_dir, input_dir, niqe_metric, brisque_metric
    )

    print('\nFinal Summary')
    print('=' * 80)
    print(f'{"Model":25s} {"NIQE":>12s} {"BRISQUE":>12s} {"LOE":>12s}')
    print('-' * 80)
    print(f'{"Zero-DCE (Epoch 8)":25s} {zero_niqe:12.4f} {zero_brisque:12.4f} {zero_loe:12.4f}')
    print(f'{"RUAS":25s} {ruas_niqe:12.4f} {ruas_brisque:12.4f} {ruas_loe:12.4f}')
    print(f'{"Retinexformer":25s} {ret_niqe:12.4f} {ret_brisque:12.4f} {ret_loe:12.4f}')
    print('=' * 80)