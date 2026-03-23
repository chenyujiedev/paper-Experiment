# -*- coding: utf-8 -*-
# @Time    : 2026/3/15 00:56
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : analyze_checkpoint_results.py

from PIL import Image
import numpy as np
import pandas as pd


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    return img


def rgb_to_luminance(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def compute_metrics(input_img, result_img, dark_threshold=0.4, bright_threshold=0.7):
    input_lum = rgb_to_luminance(input_img)
    result_lum = rgb_to_luminance(result_img)

    dark_mask = input_lum < dark_threshold
    bright_mask = input_lum > bright_threshold

    if np.sum(dark_mask) > 0:
        dark_region_mean_brightness = float(np.mean(result_lum[dark_mask]))
    else:
        dark_region_mean_brightness = np.nan

    if np.sum(bright_mask) > 0:
        highlight_region_mean_change = float(
            np.mean(np.abs(result_lum[bright_mask] - input_lum[bright_mask]))
        )
    else:
        highlight_region_mean_change = np.nan

    global_mean_brightness = float(np.mean(result_lum))

    return {
        "dark_region_mean_brightness": dark_region_mean_brightness,
        "highlight_region_mean_change": highlight_region_mean_change,
        "global_mean_brightness": global_mean_brightness,
    }


def analyze_one_group(input_path, result_dict, output_csv, dark_threshold=0.4, bright_threshold=0.7):
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
    df = df.round(4)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")
    print(df)


if __name__ == "__main__":
    # 场景1：一般低光
    input_path_1 = "../data/test_data/DICM/42.jpg"
    result_dict_1 = {
        "baseline": "../data/result_epoch8_line/DICM/42.jpg",
        "ep7": "../data/result_w3_epoch7/DICM/42.jpg",
        "ep8": "../data/result_w3_epoch8/DICM/42.jpg",
        "ep9": "../data/result_w3_epoch9/DICM/42.jpg",
    }

    # 场景2：极暗场景
    input_path_2 = "../data/test_data/DICM/01.jpg"
    result_dict_2 = {
        "baseline": "../data/result_epoch8_line/DICM/01.jpg",
        "ep7": "../data/result_w3_epoch7/DICM/01.jpg",
        "ep8": "../data/result_w3_epoch8/DICM/01.jpg",
        "ep9": "../data/result_w3_epoch9/DICM/01.jpg",
    }

    # 场景3：复杂光照
    input_path_3 = "../data/test_data/DICM/37.jpg"
    result_dict_3 = {
        "baseline": "../data/result_epoch8_line/DICM/37.jpg",
        "ep7": "../data/result_w3_epoch7/DICM/37.jpg",
        "ep8": "../data/result_w3_epoch8/DICM/37.jpg",
        "ep9": "../data/result_w3_epoch9/DICM/37.jpg",
    }

    analyze_one_group(input_path_1, result_dict_1, "../csv/ckpt_scene1.csv")
    analyze_one_group(input_path_2, result_dict_2, "../csv/ckpt_scene2.csv")
    analyze_one_group(input_path_3, result_dict_3, "../csv/ckpt_scene3.csv")