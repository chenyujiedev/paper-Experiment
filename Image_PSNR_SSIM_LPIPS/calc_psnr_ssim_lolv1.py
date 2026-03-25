# -*- coding: utf-8 -*-
# @Time    : 2026/3/16 01:34
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : calc_psnr_ssim_lolv1.py
import os
import cv2
import math
import numpy as np
from glob import glob
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim


def collect_image_paths(folder):
    return natsorted(
        glob(os.path.join(folder, '*.png')) +
        glob(os.path.join(folder, '*.jpg')) +
        glob(os.path.join(folder, '*.jpeg')) +
        glob(os.path.join(folder, '*.bmp'))
    )


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    # 转成 RGB 后逐通道算平均
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        scores = []
        for i in range(3):
            score = ssim(img1[:, :, i], img2[:, :, i], data_range=255)
            scores.append(score)
        return np.mean(scores)
    else:
        return ssim(img1, img2, data_range=255)


def evaluate_folder(pred_dir, gt_dir, model_name='model'):
    pred_paths = collect_image_paths(pred_dir)
    gt_paths = collect_image_paths(gt_dir)

    if len(pred_paths) == 0:
        raise FileNotFoundError(f'No prediction images found in: {pred_dir}')
    if len(gt_paths) == 0:
        raise FileNotFoundError(f'No GT images found in: {gt_dir}')
    if len(pred_paths) != len(gt_paths):
        raise ValueError(
            f'Image count mismatch: pred={len(pred_paths)}, gt={len(gt_paths)}'
        )

    psnr_list = []
    ssim_list = []

    print(f'\nEvaluating {model_name}')
    print(f'Prediction folder: {pred_dir}')
    print(f'GT folder        : {gt_dir}')
    print(f'Number of images : {len(pred_paths)}')

    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred_name = os.path.basename(pred_path)
        gt_name = os.path.basename(gt_path)

        pred = cv2.imread(pred_path)
        gt = cv2.imread(gt_path)

        if pred is None:
            raise ValueError(f'Failed to read prediction image: {pred_path}')
        if gt is None:
            raise ValueError(f'Failed to read GT image: {gt_path}')

        # BGR -> RGB
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        if pred.shape != gt.shape:
            raise ValueError(
                f'Shape mismatch: {pred_name} {pred.shape} vs {gt_name} {gt.shape}'
            )

        psnr_value = calculate_psnr(pred, gt)
        ssim_value = calculate_ssim(pred, gt)

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

        print(f'{pred_name:20s}  PSNR={psnr_value:.4f}  SSIM={ssim_value:.4f}')

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))

    print('-' * 60)
    print(f'{model_name} Average PSNR: {avg_psnr:.4f}')
    print(f'{model_name} Average SSIM: {avg_ssim:.4f}')
    print('-' * 60)

    return avg_psnr, avg_ssim


if __name__ == '__main__':
    gt_dir = 'data/LOLv1/Test/target'

    zero_dir = 'data/3_22/2'
    ruas_dir = 'data/result_ruas_v1'
    retinex_dir = 'data/result_retinex_v1'

    zero_psnr, zero_ssim = evaluate_folder(zero_dir, gt_dir, 'Zero-DCE (Epoch 8)')
    ruas_psnr, ruas_ssim = evaluate_folder(ruas_dir, gt_dir, 'RUAS')
    ret_psnr, ret_ssim = evaluate_folder(retinex_dir, gt_dir, 'Retinexformer')

    print('\nFinal Summary')
    print('=' * 60)
    print(f'{"Model":25s} {"PSNR":>12s} {"SSIM":>12s}')
    print('-' * 60)
    print(f'{"Zero-DCE (Epoch 8)":25s} {zero_psnr:12.4f} {zero_ssim:12.4f}')
    print(f'{"RUAS":25s} {ruas_psnr:12.4f} {ruas_ssim:12.4f}')
    print(f'{"Retinexformer":25s} {ret_psnr:12.4f} {ret_ssim:12.4f}')
    print('=' * 60)