# -*- coding: utf-8 -*-
# @Time    : 2026/3/25 23:33
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : calc_lpips.py
#python calc_lpips.py --pred_dir data/result_v1_epoch8 --gt_dir data/LOLv1/Test/target --ext png
#python calc_lpips.py --pred_dir data/result_improved_v1_epoch8 --gt_dir data/LOLv1/Test/target --ext png
#python calc_lpips.py --pred_dir data/result_retinex_v1 --gt_dir data/LOLv1/Test/target --ext png
#python calc_lpips.py --pred_dir data/result_ruas_v1 --gt_dir data/LOLv1/Test/target --ext png
import os
import glob
import argparse
from PIL import Image

import torch
import lpips
from torchvision import transforms


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_image(path, device):
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    # LPIPS expects input in [-1, 1]
    tensor = tensor * 2.0 - 1.0
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="folder of predicted/enhanced images")
    parser.add_argument("--gt_dir", type=str, required=True, help="folder of ground-truth images")
    parser.add_argument("--ext", type=str, default="png", help="image extension: png / jpg / jpeg")
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, f"*.{args.ext}")))
    if len(pred_paths) == 0:
        raise RuntimeError(f"No .{args.ext} images found in {args.pred_dir}")

    scores = []

    with torch.no_grad():
        for pred_path in pred_paths:
            name = os.path.basename(pred_path)
            gt_path = os.path.join(args.gt_dir, name)

            if not os.path.exists(gt_path):
                print(f"Skip: GT not found for {name}")
                continue

            pred = load_image(pred_path, device)
            gt = load_image(gt_path, device)

            if pred.shape != gt.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: pred {tuple(pred.shape)}, gt {tuple(gt.shape)}"
                )

            score = loss_fn(pred, gt)
            scores.append(score.item())

    if len(scores) == 0:
        raise RuntimeError("No valid image pairs were found.")

    mean_lpips = sum(scores) / len(scores)

    print("\n===== LPIPS Result =====")
    print("Pairs:", len(scores))
    print(f"Mean LPIPS: {mean_lpips:.6f}")
    print("Note: lower is better.")


if __name__ == "__main__":
    main()