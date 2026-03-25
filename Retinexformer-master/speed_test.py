# -*- coding: utf-8 -*-
# @Time    : 2026/3/25 21:45
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : speed_test.py
# python speed_test.py --opt Options/RetinexFormer_LOL_v1.yml
# -*- coding: utf-8 -*-
import logging
import time
import numpy as np
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_inference(model, opt):
    window_size = opt['val'].get('window_size', 0)
    if window_size and window_size > 0:
        model.pad_test(window_size)
    else:
        model.nonpad_test()


def main():
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"benchmark_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        # 只读取 val/test，不读取 train
        if phase not in ['val', 'test', 'test_1', 'test_2']:
            continue

        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed']
        )
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    model = create_model(opt)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Benchmarking {test_set_name}...')

        times = []
        warmup = 5
        data_list = list(test_loader)

        # warm-up
        for data in data_list[:min(warmup, len(data_list))]:
            model.feed_data(data)
            sync_device(device)
            run_inference(model, opt)
            sync_device(device)

        # benchmark
        for data in data_list:
            model.feed_data(data)

            sync_device(device)
            start = time.perf_counter()
            run_inference(model, opt)
            sync_device(device)
            end = time.perf_counter()

            _ = model.get_current_visuals()
            times.append((end - start) * 1000.0)

        avg_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        fps = 1000.0 / avg_ms

        logger.info(f'===== Retinexformer Benchmark: {test_set_name} =====')
        logger.info(f'Images: {len(times)}')
        logger.info(f'Average inference time: {avg_ms:.3f} ms/image')
        logger.info(f'Std inference time: {std_ms:.3f} ms/image')
        logger.info(f'FPS: {fps:.3f}')

        print(f'\n===== Retinexformer Benchmark: {test_set_name} =====')
        print(f'Images: {len(times)}')
        print(f'Average inference time: {avg_ms:.3f} ms/image')
        print(f'Std inference time:     {std_ms:.3f} ms/image')
        print(f'FPS:                    {fps:.3f}')


if __name__ == '__main__':
    main()