# Adapted from https://github.com/NVIDIA/RULER/blob/main/scripts/eval/evaluate.py

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse
import numpy as np

import re
import string

import jieba
from fuzzywuzzy import fuzz

from collections import Counter
from rouge import Rouge
import random


def string_match_part(preds, refs):
    score = sum([
        max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
        for pred, ref in zip(preds, refs)
    ]) / len(preds) * 100
    return round(score, 2)


def string_match_all(preds, refs):
    score = sum([
        sum([1.0 if r.lower() in pred.lower() else 0.0
             for r in ref]) / len(ref) for pred, ref in zip(preds, refs)
    ]) / len(preds) * 100
    return round(score, 2)


dataset2metric = {
    'niah': {
        'metric_fn': string_match_all,
    },
    'vt': {
        'metric_fn': string_match_all,
    },
    'cwe': {
        'metric_fn': string_match_all,
    },
    'fwe': {
        'metric_fn': string_match_all
    },
    'qa': {
        'metric_fn': string_match_part,
    },
}


def postprocess_pred(predict_str: str):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    return parser.parse_args(args)


def scorer(dataset, predictions, answers):
    dataset_type = dataset.split('_')[0]
    if len(answers) > 0 and answers[0][0] is not None:
        task_score = dataset2metric[dataset_type]['metric_fn'](predictions,
                                                               answers)
    else:
        task_score = 0.0

    return round(task_score, 2)


if __name__ == '__main__':
    args = parse_args()
    scores = dict()

    path = args.model
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers = [], []
        dataset = filename.split('.')[0]
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(postprocess_pred(data["pred"]))
                answers.append(data["answers"])
        score = scorer(dataset, predictions, answers)
        scores[dataset] = score

        print(f"{dataset}: {score}")

    out_path = os.path.join(path, "result.json")

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
