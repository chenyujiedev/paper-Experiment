# -*- coding: utf-8 -*-
# @Time    : 2026/3/25 21:45
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : speed_test.py
#python speed_test.py --ckpt snapshots/Epoch8.pth --data_path data/train_data
#python speed_test.py --ckpt snapshots_w3/Epoch1.pth --data_path data/train_data
import os
import glob
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import model_mac


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_image(image_path, device, resize=None):
    img = Image.open(image_path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    if resize is not None:
        h, w = resize
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    return x


def benchmark(model, image_paths, device, resize=None, warmup=5):
    model.eval()
    times = []

    with torch.no_grad():
        # warm-up
        for p in image_paths[:min(warmup, len(image_paths))]:
            x = load_image(p, device, resize)
            sync_device(device)
            _ = model(x)
            sync_device(device)

        # benchmark
        for p in image_paths:
            x = load_image(p, device, resize)

            sync_device(device)
            start = time.perf_counter()
            _ = model(x)
            sync_device(device)
            end = time.perf_counter()

            times.append((end - start) * 1000.0)  # ms

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / avg_ms
    return avg_ms, std_ms, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument("--data_path", type=str, required=True, help="folder of test images")
    parser.add_argument("--resize_h", type=int, default=0, help="resize height, 0 means original")
    parser.add_argument("--resize_w", type=int, default=0, help="resize width, 0 means original")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    model = model_mac.enhance_net_nopool().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    image_paths = (
        glob.glob(os.path.join(args.data_path, "*.png")) +
        glob.glob(os.path.join(args.data_path, "*.jpg")) +
        glob.glob(os.path.join(args.data_path, "*.jpeg"))
    )
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError("No images found.")

    resize = None
    if args.resize_h > 0 and args.resize_w > 0:
        resize = (args.resize_h, args.resize_w)
        print(f"Resize for benchmark: {args.resize_h} x {args.resize_w}")
    else:
        print("Resize for benchmark: original resolution")

    avg_ms, std_ms, fps = benchmark(model, image_paths, device, resize=resize, warmup=5)

    print("\n===== Zero-DCE Benchmark Result =====")
    print("Images:", len(image_paths))
    print(f"Average inference time: {avg_ms:.3f} ms/image")
    print(f"Std inference time:     {std_ms:.3f} ms/image")
    print(f"FPS:                    {fps:.3f}")


if __name__ == "__main__":
    main()