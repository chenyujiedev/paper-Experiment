# -*- coding: utf-8 -*-
# @Time    : 2026/3/14 23:30
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : analyze_tau_results.py

from PIL import Image
import numpy as np
import pandas as pd


def load_image(path):
    """
    读取图片并转成 [0,1] 范围的 float32 RGB
    """
    img = Image.open(path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    return img


def rgb_to_luminance(img):
    """
    计算亮度图 Y = 0.299R + 0.587G + 0.114B
    img: H x W x 3
    return: H x W
    """
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def compute_metrics(input_img, result_img, dark_threshold=0.4, bright_threshold=0.7):
    """
    根据原图 input_img 生成暗区和亮区 mask，
    然后在 result_img 上统计三类指标。
    """
    input_lum = rgb_to_luminance(input_img)
    result_lum = rgb_to_luminance(result_img)

    # 暗区 mask：原图中亮度低于 dark_threshold 的位置
    dark_mask = input_lum < dark_threshold

    # 亮区 mask：原图中亮度高于 bright_threshold 的位置
    bright_mask = input_lum > bright_threshold

    # 1. 暗区平均亮度（增强后）
    if np.sum(dark_mask) > 0:
        dark_region_mean_brightness = float(np.mean(result_lum[dark_mask]))
    else:
        dark_region_mean_brightness = np.nan

    # 2. 高亮区平均变化（增强后与原图的绝对差）
    if np.sum(bright_mask) > 0:
        highlight_region_mean_change = float(
            np.mean(np.abs(result_lum[bright_mask] - input_lum[bright_mask]))
        )
    else:
        highlight_region_mean_change = np.nan

    # 3. 全图平均亮度
    global_mean_brightness = float(np.mean(result_lum))

    return {
        "dark_region_mean_brightness": dark_region_mean_brightness,
        "highlight_region_mean_change": highlight_region_mean_change,
        "global_mean_brightness": global_mean_brightness,
    }


def analyze_one_group(input_path, result_dict, dark_threshold=0.4, bright_threshold=0.7):
    """
    对一个场景的一组图片做分析
    input_path: 原图路径
    result_dict: {
        "baseline": path1,
        "tau_0.4": path2,
        "tau_0.5": path3,
        "tau_0.6": path4,
    }
    """
    input_img = load_image(input_path)

    rows = []
    for name, path in result_dict.items():
        result_img = load_image(path)
        metrics = compute_metrics(
            input_img=input_img,
            result_img=result_img,
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold
        )
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # =========================
    # 你把下面路径改成你自己的
    # =========================

    # 场景1：一般低光
    input_path_1 = "../data/test_data/DICM/42.jpg"
    result_dict_1 = {
        "baseline": "data/result_epoch8_line/DICM/42.jpg",
        "tau_0.4": "data/result_tau04_epoch8/DICM/42.jpg",
        "tau_0.5": "data/result_tau05_epoch8/DICM/42.jpg",
        "tau_0.6": "data/result_tau06_epoch8/DICM/42.jpg",
    }

    # 场景2：极暗场景
    input_path_2 = "../data/test_data/DICM/01.jpg"
    result_dict_2 = {
        "baseline": "data/result_epoch8_line/DICM/01.jpg",
        "tau_0.4": "data/result_tau04_epoch8/DICM/01.jpg",
        "tau_0.5": "data/result_tau05_epoch8/DICM/01.jpg",
        "tau_0.6": "data/result_tau06_epoch8/DICM/01.jpg",
    }

    # 场景3：复杂光照
    input_path_3 = "../data/test_data/DICM/37.jpg"
    result_dict_3 = {
        "baseline": "data/result_epoch8_line/DICM/37.jpg",
        "tau_0.4": "data/result_tau04_epoch8/DICM/37.jpg",
        "tau_0.5": "data/result_tau05_epoch8/DICM/37.jpg",
        "tau_0.6": "data/result_tau06_epoch8/DICM/37.jpg",
    }

    print("\n=== Scene 1: General low-light ===")
    df1 = analyze_one_group(input_path_1, result_dict_1)
    print(df1.round(4))

    print("\n=== Scene 2: Very dark ===")
    df2 = analyze_one_group(input_path_2, result_dict_2)
    print(df2.round(4))

    print("\n=== Scene 3: Complex lighting ===")
    df3 = analyze_one_group(input_path_3, result_dict_3)
    print(df3.round(4))

    # 如果你想导出成 Excel / CSV
    df1.to_csv("scene1_metrics.csv", index=False)
    df2.to_csv("scene2_metrics.csv", index=False)
    df3.to_csv("scene3_metrics.csv", index=False)

    print("\nSaved CSV files:")
    print("scene1_metrics.csv")
    print("scene2_metrics.csv")
    print("scene3_metrics.csv")