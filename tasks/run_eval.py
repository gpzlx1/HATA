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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def first_int_match(prediction):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def recall_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    # precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return recall


def needle_score(prediction, ground_truth, **kwargs):
    prediction = normalize_answer(prediction).split(" ")
    ground_truth = normalize_answer(ground_truth).split(" ")
    common = Counter(prediction) & Counter(ground_truth)
    return sum(common.values()) / len(ground_truth)


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [
        normalize_zh_answer(token) for token in prediction_tokens
    ]
    ground_truth_tokens = [
        normalize_zh_answer(token) for token in ground_truth_tokens
    ]
    prediction_tokens = [
        token for token in prediction_tokens if len(token) > 0
    ]
    ground_truth_tokens = [
        token for token in ground_truth_tokens if len(token) > 0
    ]
    return f1_score(prediction_tokens, ground_truth_tokens)


def kv_retrieval_score(pred, label, **kwargs) -> bool:
    for c in ['\n', ':', '\"', '\'', '.', ',', '?', '!', '{', '}']:
        pred = pred.replace(c, ' ')
    words = pred.split()
    return label in words


def number_equal_score(pred, label, **kwargs) -> bool:
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label, **kwargs) -> bool:
    return label == first_int_match(pred)


def code_run_score(pred, label, **kwargs) -> bool:
    """
    Returns the score of one example in Code.Run.
    """
    pred = pred.strip()
    for c in ["\n", ".", "`", "'", '"', ":"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if len(words) == 0:
        return False
    try:
        pred = int(words[-1])
        return label == pred
    except Exception:
        return False


def code_debug_score(pred, label, **kwargs) -> bool:
    """
    Returns the score of one example in Code.Debug.
    """
    label_c = label[1]
    fn_name = label[0]
    if pred[:2] in [f"{label_c}.", f"{label_c}:"]:
        return True

    ans_prefixes = [
        "answer is:",
        "is:",
        "answer:",
    ]

    ans_prefixes_2 = [
        "answer is",
        "error is",
    ]

    pred = pred.strip()
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")

    ret = None
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            ret = False
            break
        pred = pred[idx + len(prefix) + 1 :]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                ret = True
                break
        if ret is not None:
            break
        ret = False
        break

    ret1 = ret
    ret = None

    for prefix2 in ans_prefixes_2:
        idx = pred.find(prefix2)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix2) + 1:
            ret = False
            break
        pred = pred[idx + len(prefix2) + 1 :]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                ret = True
                break
        if ret is not None:
            break
        ret = False
        break

    ret2 = ret
    if ret1 is None and ret2 is None:
        random.seed(fn_name)
        ans = random.choice(["A", "B", "C", "D"])
        # print(ans, label_c)
        if ans == label_c:
            return True
        else:
            return False
    if ret1 is None: ret1 = False
    if ret2 is None: ret2 = False
    return ret1 or ret2


def math_score(pred, label, **kwargs) -> bool:
    if isinstance(label, list):
        # In math_find, there is always only one label.
        label = label[0]
    if isinstance(label, int):
        # Find first int or float
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        return int(first_num) == label
    elif isinstance(label, float):
        # Find first float or int
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        return float(first_float) == label
    else:
        raise TypeError(f"Expected int or float, got {type(label)}")


def longdialogue_qa_eng_score(pred, label, **kwargs) -> bool:
    # label = label[0]
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    words = [x.upper() for x in words]
    # print(label, words)
    return label in words


def longbook_choice_score(pred, label, **kwargs) -> bool:
    # Just use the first letter as the prediction
    if pred[0] in "ABCD":
        return pred[0] == label
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word == label
    return False


def math_calc_score(pred, label, **kwargs) -> float:
    assert isinstance(label, list), f"Expected list, got {type(label)}"
    # assert isinstance(pred, list), f"Expected list, got {type(pred)}"
    pred_nums = []
    pred_list = re.split("[^0-9]", pred)
    for item in pred_list:
        if item != "":
            pred_nums.append(int(item))

    # Our prompts makes GPT4 always output the first number as the first value
    # in the predicted answer.
    # if model_name == "gpt4":
    #     pred_nums = pred_nums[1:]

    cnt = 0
    for i in range(len(label)):
        if i >= len(pred_nums):
            break
        if label[i] == pred_nums[i]:
            cnt += 1
        else:
            break
    return cnt / len(label)


dataset2metric = {
    # longbench
    "narrativeqa": qa_f1_score,
    "narrativeqa_retrieval": needle_score,
    "qasper": qa_f1_score,
    "qasper_retrieval": needle_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_en_retrieval": needle_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "hotpotqa_retrieval": needle_score,
    "2wikimqa": qa_f1_score,
    "2wikimqa_retrieval": needle_score,
    "musique": qa_f1_score,
    "musique_retrieval": needle_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "gov_report_retrieval": needle_score,
    "qmsum": rouge_score,
    "qmsum_retrieval": needle_score,
    "multi_news": rouge_score,
    "multi_news_retrieval": needle_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "trec_retrieval": needle_score,
    "triviaqa": qa_f1_score,
    "triviaqa_retrieval": needle_score,
    "samsum": rouge_score,
    "samsum_retrieval": needle_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_en_retrieval": needle_score,
    "passage_count": count_score,
    "passage_count_retrieval": needle_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "lcc_retrieval": needle_score,
    "repobench-p": code_sim_score,
    "repobench-p_retrieval": needle_score,

    # infinitebench
    # Retrieve
    "kv_retrieval": kv_retrieval_score,
    "kv_retrieval_prefix": kv_retrieval_score,
    "kv_retrieval_both": kv_retrieval_score,
    "passkey": number_equal_score,
    "number_string": number_equal_score,
    # Code
    "code_run": code_run_score,
    "code_debug": code_debug_score,
    # Longbook
    "longdialogue_qa_eng": longdialogue_qa_eng_score,
    "longbook_qa_eng": qa_f1_score,
    "longbook_sum_eng": rouge_score,
    "longbook_choice_eng": longbook_choice_score,
    "longbook_qa_chn": qa_f1_zh_score,
    # Math
    "math_find": math_score,
    "math_calc": math_calc_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e',
                        action='store_true',
                        help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers,
                                                   lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(
                score, dataset2metric[dataset](prediction,
                                               ground_truth,
                                               all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]

        if dataset in ["code_debug"]:
            score = dataset2metric[dataset](prediction,
                                            ground_truths,
                                            all_classes=all_classes)
        else:
            for ground_truth in ground_truths:
                score = max(
                    score, dataset2metric[dataset](prediction,
                                                ground_truth,
                                                all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()
    scores = dict()

    path = args.model
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        if "_e" in dataset[-2:]:
            dataset = dataset[:-2]
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths,
                             all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score

        print(f"{dataset}: {score}")

    out_path = os.path.join(path, "result.json")

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
