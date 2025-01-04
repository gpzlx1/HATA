import torch
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List, Dict, Any
import time
import numpy as np


def get_max_length_in_nested_lists(lst):
    if len(lst) and isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)


def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        masks = []
        for i, elem in enumerate(lst):
            lst[i], mask = pad_nested_lists(elem, max_length, padding_value,
                                            padding_side)
            masks.append(mask)
        return lst, masks
    elif isinstance(lst, list):
        if padding_side == "right":
            mask = [1] * len(lst) + [0] * (max_length - len(lst))
            lst = lst + [padding_value for _ in range(max_length - len(lst))]
            return lst, mask
        else:
            mask = [0] * (max_length - len(lst)) + [1] * len(lst)
            lst = [padding_value for _ in range(max_length - len(lst))] + lst
            return lst, mask
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")


@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """
    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    keys_to_tensorize = {
        "input_ids", "attention_mask", "labels", "position_ids",
        "token_type_ids", "depth", "index"
    }

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        first_elem = batch_elem[0]
        return_batch = {}

        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
            if isinstance(value, list) and key in self.keys_to_tensorize:
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length,
                                                  pad_token_id,
                                                  self.tokenizer.padding_side)

            if key in self.keys_to_tensorize:
                return_batch[key] = torch.tensor(batch_value)
            else:
                # handle strings and None
                return_batch[key] = batch_value
        return return_batch


class TimeRecoder():

    def __init__(self, cuda_sync=True):
        self.begins = {}
        self.ends = {}
        self.durations = {}
        self.running_mask = {}
        self.cuda_sync = cuda_sync

    def start(self, name):
        if name not in self.running_mask:
            self.begins[name] = []
            self.ends[name] = []
            self.durations[name] = []
            self.running_mask[name] = False
        assert self.running_mask[name] == False, f"Event {name} is running!"

        if self.cuda_sync:
            torch.cuda.synchronize()
        self.begins[name].append(time.time())
        self.running_mask[name] = True

    def end(self, name):
        assert name in self.running_mask, f"Event {name} is not running!"
        assert self.running_mask[name] == True, f"Event {name} is not running!"

        if self.cuda_sync:
            torch.cuda.synchronize()
        self.ends[name].append(time.time())
        self.durations[name].append(self.ends[name][-1] -
                                    self.begins[name][-1])
        self.running_mask[name] = False

    def check_all_recoder(self):
        for name in self.durations:
            print(f"{name}: {np.mean(self.durations[name][2:]) * 1000:.3f}")
