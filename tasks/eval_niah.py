################################################################################
# Copyright (c) Microsoft Corporation. and affiliates
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import json
from collections import Counter

from argparse import ArgumentParser, Namespace
import json, os
import re


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--input",
                   type=str,
                   default='preds/niah-longchat-hash128-top0.03/')
    p.add_argument("--output", type=str, default='figures')
    p.add_argument("--model", type=str, default='longchat-7b-v1.5-32k')
    p.add_argument("--method", type=str, default='HashAttention')
    return p.parse_args()


def summary(args):
    datas = []
    res = Counter()

    path = os.path.join(args.input, "niah.jsonl")

    min_context = None
    max_context = None

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())

            processed_data = {}

            processed_data["context_length"] = data["length"]
            processed_data["depth_percent"] = data["depth_percent"]

            length = int(data["length"])
            min_context = min(min_context,
                              length) if min_context is not None else length
            max_context = max(max_context,
                              length) if max_context is not None else length

            pred = data["pred"].strip()
            pred = re.split("[^0-9]", pred)[0]
            answer = data["answers"][0]
            processed_data["correct"] = pred == answer

            res[(data["length"],
                 data["depth_percent"])] += processed_data["correct"] == True

            if not processed_data["correct"]:
                print("WRONG", data)

            datas.append(processed_data)

    with open("tmp.json", "w") as json_file:
        json.dump(datas, json_file)

    return res, min_context, max_context


def plot_needle_viz(
    args,
    res_file="tmp.json",
    min_context=1024,
    max_context=1000000,
):

    def get_context_size(x):
        return f"{round(x / 1000)}K"

    plt.rc("axes", titlesize=25)  # fontsize of the title
    plt.rc("axes", labelsize=25)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=20)  # fontsize of the legend

    df = pd.read_json(res_file)
    os.system(f"rm {res_file}")
    accuracy_df = df.groupby(["context_length",
                              "depth_percent"])["correct"].mean()
    accuracy_df = accuracy_df
    accuracy_df = accuracy_df.reset_index()
    accuracy_df = accuracy_df.rename(
        columns={
            "correct": "Score",
            "context_length": "Context Length",
            "depth_percent": "Document Depth",
        })

    pivot_table = pd.pivot_table(
        accuracy_df,
        values="Score",
        index=["Document Depth", "Context Length"],
        aggfunc="mean",
    ).reset_index()
    pivot_table = pivot_table.pivot(index="Document Depth",
                                    columns="Context Length",
                                    values="Score")

    cmap = LinearSegmentedColormap.from_list("custom_cmap",
                                             ["#F0496E", "#EBB839", "#0CD79F"])

    plt.figure(figsize=(14, 7))
    ax = sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    min_context_str = f"{min_context // 1000}K" if min_context >= 1000 else min_context
    max_context_str = f"{max_context // 1000}K" if max_context >= 1000 else max_context

    # More aesthetics
    name = f" w/ {args.method}"

    plt.title(f"Needle in A Haystack {args.model}{name}")  # Adds a title
    plt.xlabel("Context Length")  # X-axis label
    plt.ylabel("Depth Percent (%)")  # Y-axis label

    # Centering x-ticks
    xtick_labels = pivot_table.columns.values
    xtick_labels = [get_context_size(x) for x in xtick_labels]
    ax.set_xticks(np.arange(len(xtick_labels)) + 0.5, minor=False)
    ax.set_xticklabels(xtick_labels)

    # Drawing white grid lines
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1)

    # Iterate over the number of pairs of gridlines you want
    for i in range(pivot_table.shape[0]):
        ax.axhline(i, color="white", lw=1)
    for i in range(pivot_table.shape[1]):
        ax.axvline(i, color="white", lw=1)

    # Ensure the ticks are horizontal and prevent overlap
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)

    # Fit everything neatly into the figure area
    plt.tight_layout()

    # Save and Show the plot
    save_path = os.path.join(
        args.output,
        f"NiaH_{args.model}_{args.method}_{min_context_str}_{max_context_str}.pdf",
    )
    plt.savefig(save_path, dpi=1000)
    print(f"Needle plot saved to {save_path}.")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    res, min_context, max_context = summary(args)
    plot_needle_viz(args, min_context=min_context, max_context=max_context)
