# -*- coding: utf-8 -*-
# @Time    : 2026/3/25 21:45
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : speed_test.py
# python speed_test.py --data_path ./data/test_data --model upe
import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.utils

from model import Network
from multi_read_data import MemoryFriendlyLoader


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser("RUAS benchmark")
    parser.add_argument('--data_path', type=str, required=True,
                        help='folder path of test images')
    parser.add_argument('--model', type=str, default='upe',
                        help='checkpoint name: upe / lol / dark')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_dataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
    test_queue = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=False,
        num_workers=0
    )

    model = Network().to(device)
    ckpt_path = './ckpt/' + args.model + '.pt'
    model_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(model_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    times = []
    warmup = 5

    with torch.no_grad():
        # warm-up
        for i, (input_tensor, image_name) in enumerate(test_queue):
            input_tensor = input_tensor.to(device)
            sync_device(device)
            _ = model(input_tensor)
            sync_device(device)
            if i + 1 >= warmup:
                break

        # benchmark
        for i, (input_tensor, image_name) in enumerate(test_queue):
            input_tensor = input_tensor.to(device)

            sync_device(device)
            start = time.perf_counter()
            u_list, r_list = model(input_tensor)
            sync_device(device)
            end = time.perf_counter()

            # choose output branch only to keep logic consistent
            if args.model == 'lol':
                _ = u_list[-1]
            elif args.model in ['upe', 'dark']:
                _ = u_list[-2]
            else:
                _ = u_list[-1]

            times.append((end - start) * 1000.0)

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / avg_ms

    print("\n===== RUAS Benchmark Result =====")
    print("Images:", len(times))
    print(f"Average inference time: {avg_ms:.3f} ms/image")
    print(f"Std inference time:     {std_ms:.3f} ms/image")
    print(f"FPS:                    {fps:.3f}")


if __name__ == '__main__':
    main()