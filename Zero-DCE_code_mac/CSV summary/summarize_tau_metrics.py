# -*- coding: utf-8 -*-
# @Time    : 2026/3/15 00:40
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : summarize_tau_metrics.py
import pandas as pd


def summarize_tau_metrics(scene_csv_list, output_csv="tau_summary.csv"):
    """
    scene_csv_list: 例如 ["scene1_metrics.csv", "scene2_metrics.csv", "scene3_metrics.csv"]
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

    # 保证顺序
    order = ["baseline", "tau_0.4", "tau_0.5", "tau_0.6"]
    summary["model"] = pd.Categorical(summary["model"], categories=order, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)

    # 保留4位小数
    summary = summary.round(4)

    summary.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print(summary)


if __name__ == "__main__":
    scene_csv_list = [
        "scene1_metrics.csv",
        "scene2_metrics.csv",
        "scene3_metrics.csv",
    ]
    summarize_tau_metrics(scene_csv_list, output_csv="../csv/tau_summary.csv")