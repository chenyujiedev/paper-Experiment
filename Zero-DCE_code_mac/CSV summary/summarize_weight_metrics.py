# -*- coding: utf-8 -*-
# @Time    : 2026/3/15 00:40
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : summarize_weight_metrics.py

import pandas as pd


def summarize_weight_metrics(scene_csv_list, output_csv="weight_summary.csv"):
    """
    scene_csv_list: 例如 ["weight_scene1.csv", "weight_scene2.csv", "weight_scene3.csv"]
    output_csv: 汇总输出文件
    """
    dfs = []
    for csv_path in scene_csv_list:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    summary = (
        all_df.groupby("model", as_index=False)[
            [
                "dark_region_mean_brightness",
                "highlight_region_mean_change",
                "global_mean_brightness",
            ]
        ]
        .mean()
    )

    order = ["baseline", "w1", "w3", "w5"]
    summary["model"] = pd.Categorical(summary["model"], categories=order, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)

    summary = summary.round(4)

    summary.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print(summary)


if __name__ == "__main__":
    scene_csv_list = [
        "../csv/weight_scene1.csv",
        "../csv/weight_scene2.csv",
        "../csv/weight_scene3.csv",
    ]
    summarize_weight_metrics(scene_csv_list, output_csv="../csv/weight_summary.csv")