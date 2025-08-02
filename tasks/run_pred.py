import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from dataset_utils import (
    DefaultDataCollator,
    GetManagerAndTasks
)

from model_utils import (
    load_config_and_tokenizer,
    load_model,
    comm_generate,
)

import torch.multiprocessing as mp


import signal

timeout = 0


def handler(signum, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, handler)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def pred_loop_func(args, rank, task_queue, dataset_manager):
    seed_everything(args.seed)

    args.method = args.method.lower()

    model_meta, model_config, tokenizer, _, _ = load_config_and_tokenizer(
        args, args.model_name_or_path)
    model = load_model(model_meta, model_config, args.model_name_or_path)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    device_ids = [args.pp_num * rank + i for i in range(args.pp_num)]
    if len(device_ids) > 1:
        max_memory = {}
        for id in device_ids:
            max_memory[id] = torch.cuda.mem_get_info(id)[0]
        map_kwargs = {"max_memory": get_balanced_memory(model, max_memory)}
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=[
                                               "CustomLlamaDecoderLayer",
                                               "CustomGlmDecoderLayer",
                                               "CustomGLMBlock",
                                               "CustomQwen2DecoderLayer"
                                           ],
                                           **map_kwargs)
        model = dispatch_model(model, device_map=device_map)
    else:
        device = f"cuda:{device_ids[0]}"
        torch.cuda.set_device(device)
        model = model.to(device)

    for id in device_ids:
        print(
            f"Worker {rank} use GPU[{id}] mem: {torch.cuda.max_memory_allocated(id) / 1024 ** 3:.2f} GB"
        )

    try:
        signal.alarm(timeout)  # start timer
        while True:
            data = task_queue.get()
            if data is None:
                break
            warm_up, dataset_name, x, generate_kwargs = data
            indices = x.pop("index").tolist()

            out_info = x.copy()

            x["input_ids"] = x["input_ids"].to(model.device)
            x["attention_mask"] = x["attention_mask"].to(model.device)

            preserve_domins = ["input_ids", "attention_mask"]

            for key in out_info:
                num_obj = len(out_info[key])
                if key not in preserve_domins:
                    x.pop(key)

            for key in preserve_domins:
                out_info.pop(key)

            outputs = comm_generate(x, generate_kwargs, model, tokenizer)

            if not warm_up:
                for i in range(num_obj):
                    dataset_manager.write_one_result_v3(
                        args.output_dir, outputs[i], i, out_info, dataset_name)

            # reset timer
            signal.alarm(timeout)

        signal.alarm(0)
    except TimeoutError:
        print("The operation timed out. Exiting!")

    try:
        if args.breakdown:
            for l in range(len(model.model.layers)):
                print(f"====== Layer {l:2d} Time Log ======")
                print("Time units: ms")
                print(
                    f"Total prefill times: {model.model.layers[l].self_attn.req_num}"
                )
                model.model.layers[l].self_attn.timer.check_all_recoder()
    except:
        print(f"Breakdown not implemented for {args.method}!")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_maxlen", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="longbench",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--e",
                        action="store_true",
                        help="Evaluate on LongBench-E")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--method", type=str, default="flashattn")
    parser.add_argument("--write_in_time", action="store_true")
    parser.add_argument("--mp_num", default=1, type=int)
    parser.add_argument("--pp_num", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--min_seq_len", type=int, default=16000)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    print(args)
    seed_everything(args.seed)

    dataset_manager, tasks = GetManagerAndTasks(
        args.dataset_name, args.dataset_path, args.e)
    print("datasets: ", tasks)

    # load model and tokenizer
    args.method = args.method.lower()
    model_meta, model_config, tokenizer, generate_kwargs, apply_chat_template = load_config_and_tokenizer(
        args, args.model_name_or_path)

    # set generate_kwargs
    if hasattr(model_config, "eos_token_id"):
        eos_token_id = model_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    exit()

    # start processes
    task_queue = mp.Queue(maxsize=args.mp_num)
    work_processes = []
    for i in range(args.mp_num):
        p = mp.Process(
            target=pred_loop_func,
            args=(args, i, task_queue, dataset_manager),
        )
        p.start()
        work_processes.append(p)

    warm_up = True

    for dataset_name in tasks:
        raw_data = dataset_manager.get_data(dataset_name)
        _, dataset_maxlen, dataset_category = dataset_manager.get_dataset_info(
            dataset_name)

        process_fn = partial(
            dataset_manager.process_raw_data,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            task=dataset_name,
            max_length=args.model_maxlen,
            truncate_from_middle=True,
        )

        remove_columns = []
        for key in raw_data[0]:
            if key not in [
                    "length",
                    "all_classes",
                    "answers",
                    "depth_percent",
                    "difficulty",
                    "domain",
                    "sub_domain",
                    "answer",
                    "canonical_solution",
                    "test",
                    "entry_point",
                    "answerKey",
            ]:
                remove_columns.append(key)
        encoded_data = raw_data.map(
            process_fn,
            batched=True,
            num_proc=4,
            batch_size=10,
            with_indices=True,
            remove_columns=remove_columns,
        )

        dataloader = torch.utils.data.DataLoader(
            encoded_data,
            batch_size=args.batch_size,
            collate_fn=DefaultDataCollator(tokenizer=tokenizer),
            pin_memory=False,
        )

        generate_kwargs["max_new_tokens"] = dataset_maxlen
        if dataset_name in [
                "2wikimqa",
                "hotpotqa",
                "musique",
                "multifieldqa_en",
                "qasper",
                "narrativeqa",
                "samsum",
        ]:
            generate_kwargs["eos_token_id"] = eos_token_id
            if dataset_category is not None and "QA" in dataset_category:
                generate_kwargs["eos_token_id"].append(
                    tokenizer.encode("\n", add_special_tokens=False)[-1])
        else:
            generate_kwargs.pop("eos_token_id", None)

        if warm_up:
            for i, x in enumerate(tqdm(dataloader, desc=f"Warm up")):
                if x["input_ids"].size(-1) < args.min_seq_len:
                    continue
                for _ in range(args.mp_num):
                    task_queue.put((warm_up, dataset_name, x, generate_kwargs))
                warm_up = False
                break

        if warm_up:
            continue

        for i, x in enumerate(
                tqdm(dataloader, desc=f"Send tasks for {dataset_name}")):
            if x["input_ids"].size(-1) < args.min_seq_len:
                continue
            task_queue.put((warm_up, dataset_name, x, generate_kwargs))

    for _ in range(args.mp_num):
        task_queue.put(None)

    for p in work_processes:
        p.join()
