# -*- coding: utf-8 -*-
# @Time    : 2026/3/15 00:58
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : summarize_checkpoint_metrics.py
import pandas as pd


def summarize_checkpoint_metrics(scene_csv_list, output_csv="checkpoint_summary.csv"):
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

    order = ["baseline", "ep7", "ep8", "ep9"]
    summary["model"] = pd.Categorical(summary["model"], categories=order, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)

    summary = summary.round(4)
    summary.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(summary)


if __name__ == "__main__":
    scene_csv_list = [
        "../csv/ckpt_scene1.csv",
        "../csv/ckpt_scene2.csv",
        "../csv/ckpt_scene3.csv",
    ]
    summarize_checkpoint_metrics(scene_csv_list, output_csv="../csv/checkpoint_summary.csv")