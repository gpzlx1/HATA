import os
import json
from functools import partial

import torch
from datasets import load_dataset, Dataset

from utils import DefaultDataCollator

datasets_prompt = {
    "narrativeqa":
    "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper":
    'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en":
    "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh":
    "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa":
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa":
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique":
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader":
    "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report":
    "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum":
    "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news":
    "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum":
    "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec":
    "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa":
    "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum":
    "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht":
    "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count":
    "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en":
    'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh":
    '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc":
    "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p":
    "Please complete the code given below. \n{context}{input}Next line of code:\n",

    "narrativeqa_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "qasper_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "multifieldqa_en_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "hotpotqa_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "2wikimqa_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "musique_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "dureader_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "gov_report_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "qmsum_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "multi_news_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "trec_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "triviaqa_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "samsum_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "lsht_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "passage_count_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "passage_retrieval_en_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "lcc_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "repobench-p_retrieval":
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",

    # InfiniteBench
    "passkey": 
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "all_passkey": 
    "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "number_string": 
    "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}",
    "kv_retrieval": 
    "Extract the value corresponding to the specified key {key} in the JSON object below.\n\n{context}\n\n{input}", 
    "longbook_qa_eng": 
    "Read the book below and answer a question.\n\n{context}\n\nQuestion: {input}\n\nPlease answer as short as possible. The answer is:", 
    "longbook_qa_eng_question_first": 
    "Read the book below and answer the question.\n\nQuestion: {input}\n\n{context}\n\nQuestion: {input}\n\nPlease answer as short as possible. The answer is:", 
    "longbook_choice_eng": 
    "Read the book and answer the question.\n\n{context}\n\nQuestion: {input}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",
    "longbook_sum_eng": 
    "Summarize the following book.\n\n{context}",
    "longbook_qa_chn": 
    "请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{input}\n请尽量简短地回答。",
    "math_find": 
    "{prefix}\n\n{context}\n\n{input}",
    "math_calc": 
    "Compute the intermediate values in the following long expression.\n\n{context}",
    "code_run": 
    "Following is a set of Python functions. There is a function called named {func}.\n\n{context}\n\nPlease give me the exact number of the return value of {func_call}. Be concise. Your response must end with the final returned value.",
    "code_debug": 
    "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nGive me your answer for the function that has the deliberate and obvious error in A, B, C, or D. Your answer MUST be chosen from one of the four options without any explanation. If you cannot determine answers accurately, you also MUST provide the answer you think is most likely. Absolutely do not say you do not know or you need more information.", 
    "longdialogue_qa_eng": 
    "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else."

}

datasets_maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,

    "narrativeqa_retrieval": 64,
    "qasper_retrieval": 64,
    "multifieldqa_en_retrieval": 64,
    "hotpotqa_retrieval": 64,
    "2wikimqa_retrieval": 64,
    "musique_retrieval": 64,
    "gov_report_retrieval": 64,
    "qmsum_retrieval": 64,
    "multi_news_retrieval": 64,
    "trec_retrieval": 64,
    "triviaqa_retrieval": 64,
    "samsum_retrieval": 64,
    "passage_count_retrieval": 64,
    "passage_retrieval_en_retrieval": 64,
    "lcc_retrieval": 64,
    "repobench-p_retrieval": 64,

    # InfiniteBench
    "passkey": 12,
    "number_string": 32,
    "kv_retrieval": 50,
    "longbook_sum_eng": 1200,
    "longbook_choice_eng": 40,
    "longbook_qa_eng": 40,
    "longbook_qa_chn": 40,
    "longdialogue_qa_eng": 40,
    "math_find": 32,
    "math_calc": 30000,
    "code_run": 32,
    "code_debug": 32,
}

datasets_category = {
    "narrativeqa": "EN Single-Doc QA",
    "qasper": "EN Single-Doc QA",
    "multifieldqa_en": "EN Single-Doc QA",
    "multifieldqa_zh": "CN Single-Doc QA",
    "hotpotqa": "EN Multi-Doc QA",
    "2wikimqa": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "dureader": "CN Multi-Doc QA",
    "gov_report": "EN Summarization",
    "qmsum": "EN Summarization",
    "multi_news": "EN Summarization",
    "vcsum": "CN Summarization",
    "trec": "EN Few-Shot Learning",
    "triviaqa": "EN Few-Shot Learning",
    "samsum": "EN Few-Shot Learning",
    "lsht": "CN Few-Shot Learning",
    "passage_retrieval_en": "EN Synthetic Task",
    "passage_count": "EN Synthetic Task",
    "passage_retrieval_zh": "CN Synthetic Task",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion",

    "narrativeqa_retrieval": "EN Single-Doc QA",
    "qasper_retrieval": "EN Single-Doc QA",
    "multifieldqa_en_retrieval": "EN Single-Doc QA",
    "hotpotqa_retrieval": "EN Single-Doc QA",
    "2wikimqa_retrieval": "EN Single-Doc QA",
    "musique_retrieval": "EN Single-Doc QA",
    "gov_report_retrieval": "EN Single-Doc QA",
    "qmsum_retrieval": "EN Single-Doc QA",
    "multi_news_retrieval": "EN Single-Doc QA",
    "trec_retrieval": "EN Single-Doc QA",
    "triviaqa_retrieval": "EN Single-Doc QA",
    "samsum_retrieval": "EN Single-Doc QA",
    "passage_count_retrieval": "EN Single-Doc QA",
    "passage_retrieval_en_retrieval": "EN Single-Doc QA",
    "lcc_retrieval": "EN Single-Doc QA",
    "repobench-p_retrieval": "EN Single-Doc QA",

    "code_debug": None,
    "code_run": None,
    "passkey": None,
    "number_string": None,
    "kv_retrieval": None,
    "math_find": None,
    "math_calc": None,
    "longbook_sum_eng": None,
    "longbook_choice_eng": None,
    "longbook_qa_eng": None,
    "longbook_qa_chn": None,
    "longdialogue_qa_eng": None,
}


def load_custom_json_dataset(path, data_name):
    fin = open(os.path.join(path, data_name + ".jsonl"), "r", encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        instance = {
            "_id": eg["id"],
            "context": eg["context"],
            "input": eg["input"],
            "answers": [eg["answer"]]
        }
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None
        ret.append(instance)

    return Dataset.from_list(ret)


def load_processed_infinitebench_dataset(path, data_name):
    fin = open(os.path.join(path, data_name + ".jsonl"), "r", encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        ret.append(eg)

    return Dataset.from_list(ret)


class DatasetManager:

    def __init__(self, path, data_dir):
        self.path = path
        self.data_dir = data_dir

    @staticmethod
    def get_dataset_names():
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def get_dataset_info(self):
        raise NotImplementedError

    def write_results(self, ouput_dir, indices, preds, raw_data, dataset_name):
        if not os.path.exists(ouput_dir):
            os.makedirs(ouput_dir)
        with open(os.path.join(ouput_dir, f"{dataset_name}.jsonl"), "w", encoding="utf-8") as f:
            for i, pred in zip(indices, preds):
                json_obj = raw_data[i]
                obj = {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                }
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")

    def write_one_result(self, output_dir, pred, json_obj, dataset_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, f"{dataset_name}.jsonl"), "a", encoding="utf-8") as f:
            obj = {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    def write_one_result_v2(
        self, output_dir, pred, answer, all_classes, length, dataset_name
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(
            os.path.join(output_dir, f"{dataset_name}.jsonl"), "a", encoding="utf-8"
        ) as f:
            obj = {
                "pred": pred,
                "answers": answer,
                "all_classes": all_classes,
                "length": length,
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def process_raw_data():
        raise NotImplementedError


class LongBenchManager(DatasetManager):

    def __init__(self, path, data_dir, split, with_e=False, retrieval=False):
        super().__init__(path, data_dir)
        self.with_e = with_e
        self.retrieval = retrieval
        self.split = split

    @staticmethod
    def get_dataset_names(with_e=False, retrieval=False):
        if retrieval:
            if with_e:
                datasets = [
                    "qasper_retrieval_e",
                    "multifieldqa_en_retrieval_e",
                    "hotpotqa_retrieval_e",
                    "2wikimqa_retrieval_e",
                    "gov_report_retrieval_e",
                    "multi_news_retrieval_e",
                    "trec_retrieval_e",
                    "triviaqa_retrieval_e",
                    "samsum_retrieval_e",
                    "passage_count_retrieval_e",
                    "passage_retrieval_en_retrieval_e",
                    "lcc_retrieval_e",
                    "repobench-p_retrieval_e",
                ]
            else:
                datasets = [
                    "narrativeqa_retrieval",
                    "qasper_retrieval",
                    "multifieldqa_en_retrieval",
                    "hotpotqa_retrieval",
                    "2wikimqa_retrieval",
                    "musique_retrieval",
                    "gov_report_retrieval",
                    "qmsum_retrieval",
                    "multi_news_retrieval",
                    "trec_retrieval",
                    "triviaqa_retrieval",
                    "samsum_retrieval",
                    "passage_count_retrieval",
                    "passage_retrieval_en_retrieval",
                    "lcc_retrieval",
                    "repobench-p_retrieval",
                ]
        else:
            if with_e:
                datasets = [
                    "qasper_e",
                    "multifieldqa_en_e",
                    "hotpotqa_e",
                    "2wikimqa_e",
                    "gov_report_e",
                    "multi_news_e",
                    "trec_e",
                    "triviaqa_e",
                    "samsum_e",
                    "passage_count_e",
                    "passage_retrieval_en_e",
                    "lcc_e",
                    "repobench-p_e",
                ]
            else:
                datasets = [
                    "narrativeqa",
                    "qasper",
                    "multifieldqa_en",
                    "multifieldqa_zh",
                    "hotpotqa",
                    "2wikimqa",
                    "musique",
                    "dureader",
                    "gov_report",
                    "qmsum",
                    "multi_news",
                    "vcsum",
                    "trec",
                    "triviaqa",
                    "samsum",
                    "lsht",
                    "passage_count",
                    "passage_retrieval_en",
                    "passage_retrieval_zh",
                    "lcc",
                    "repobench-p",
                ]

        return datasets

    def get_data(self, dataset_name):
        if self.retrieval:
            data = load_custom_json_dataset(
                self.data_dir,
                dataset_name)
        else:
            data = load_dataset(self.path,
                                dataset_name,
                                data_dir=self.data_dir,
                                split=self.split)
        return data

    def get_dataset_info(self, dataset_name):
        if self.with_e:
            dataset_name = dataset_name[:-2]
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            datasets_category[dataset_name],
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        if task.endswith("_e"):
            task = task[:-2]

        for input, context, index in zip(data["input"], data["context"],
                                         indices):
            prompt_template = datasets_prompt[task]
            prompt = prompt_template.format(input=input, context=context)

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            # in fewshot learning and code completion we do not need chat template
            if datasets_category[task] is None or not any(x in datasets_category[task]
                       for x in ["Few-Shot Learning", "Code Completion"]):
                encoded = apply_chat_template(prompt, tokenizer)

            else:
                encoded = tokenizer(prompt)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


class InfiniteBenchManager(DatasetManager):

    def __init__(self, path, data_dir):
        super().__init__(path, data_dir)

    @staticmethod
    def get_dataset_names():
        datasets = [
            "code_debug",
            "code_run",
            "passkey",
            "number_string",
            "kv_retrieval",
            "math_find",
            "math_calc",
            "longbook_sum_eng",
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_qa_chn",
            "longdialogue_qa_eng",
        ]

        return datasets

    def get_data(self, dataset_name):
        return load_processed_infinitebench_dataset(self.data_dir, dataset_name)

    def get_dataset_info(self, dataset_name):
        return (
            datasets_prompt[dataset_name],
            datasets_maxlen[dataset_name],
            datasets_category[dataset_name],
        )

    @staticmethod
    def process_raw_data(
        data,
        indices,
        tokenizer,
        apply_chat_template,
        task,
        max_length=3500,
        truncate_from_middle=True,
    ):
        outputs = {"input_ids": [], "attention_mask": [], "index": []}
        for it, index in enumerate(indices):
            prompt_template = datasets_prompt[task]
            if task == "kv_retrieval":
                prompt = prompt_template.format(
                    input=data["input"][it],
                    context=data["context"][it],
                    key=data["key"][it])
            elif task == "longbook_choice_eng":
                prompt = prompt_template.format(
                    input=data["input"][it],
                    context=data["context"][it],
                    OPTION_A=data["OPTION_A"][it],
                    OPTION_B=data["OPTION_B"][it],
                    OPTION_C=data["OPTION_C"][it],
                    OPTION_D=data["OPTION_D"][it])
            elif task in ["longbook_sum_eng", "math_calc",
                          "longdialogue_qa_eng"]:
                prompt = prompt_template.format(
                    context=data["context"][it])
            elif task == "math_find":
                prompt = prompt_template.format(
                    input=data["input"][it],
                    context=data["context"][it],
                    prefix=data["prefix"][it])
            elif task == "code_run":
                prompt = prompt_template.format(
                    context=data["context"][it],
                    func=data["func"][it],
                    func_call=data["func_call"][it])
            elif task == "code_debug":
                prompt = prompt_template.format(
                    context=data["context"][it],
                    OPTION_A=data["OPTION_A"][it],
                    OPTION_B=data["OPTION_B"][it],
                    OPTION_C=data["OPTION_C"][it],
                    OPTION_D=data["OPTION_D"][it])
            else:
                prompt = prompt_template.format(
                    input=data["input"][it],
                    context=data["context"][it])

            if truncate_from_middle:
                tokenized_prompt = tokenizer.encode(prompt)
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = tokenizer.decode(
                        tokenized_prompt[:half],
                        skip_special_tokens=True) + tokenizer.decode(
                            tokenized_prompt[-half:], skip_special_tokens=True)
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                prompt = tokenizer.decode(tokenized_prompt[-max_length:],
                                          skip_special_tokens=True)

            # in fewshot learning and code completion we do not need chat template
            if datasets_category[task] is None or not any(x in datasets_category[task]
                       for x in ["Few-Shot Learning", "Code Completion"]):
                encoded = apply_chat_template(prompt, tokenizer)

            else:
                encoded = tokenizer(prompt)

            outputs["input_ids"].append(encoded["input_ids"])
            outputs["attention_mask"].append(encoded["attention_mask"])
            outputs["index"].append(index)

        return outputs


if __name__ == "__main__":
    dataset_path = "/nfs/shared_LLM_dataset/LongBench"
    with_e = True

    dataset_manager = LongBenchManager(dataset_path,
                                       dataset_path,
                                       "test",
                                       with_e=with_e)
    task = dataset_manager.get_dataset_names(with_e)[0]
    print(task)

    raw_data = dataset_manager.get_data(task)

    from transformers import LlamaTokenizer

    model_name = "llama2-7b-chat-4k"
    model_path = "/nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf"
    model_maxlen = 3500
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = "left"

    def apply_chat_template(prompt, tokenizer):
        prompt = f"[INST] {prompt} [/INST]"
        encoded = tokenizer(prompt)
        return encoded

    process_fn = partial(
        dataset_manager.process_longbench,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        task=task,
        max_length=1000,
        truncate_from_middle=True,
    )

    encoded_data = raw_data.map(
        process_fn,
        batched=True,
        num_proc=8,
        batch_size=10,
        with_indices=True,
        remove_columns=raw_data.column_names,
    )

    all_dataset = (raw_data, encoded_data)

    data_collator = DefaultDataCollator(tokenizer=tokenizer)

    dataloader = torch.utils.data.DataLoader(encoded_data,
                                             batch_size=8,
                                             collate_fn=data_collator)

    answers = raw_data["answers"]

    print(answers)

    for x in dataloader:
        print(x)
        break
