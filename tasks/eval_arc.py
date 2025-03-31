from typing import Dict, List
import blobfile as bf
import gzip
import orjson
import json
import argparse
from typing import Any, Dict, List
import os
import re


def json_loads(s: str) -> Dict:
    try:
        return orjson.loads(s)
    except Exception:
        return json.loads(s)  # fallback


def open_jsonl(file: str):
    if file.endswith(".gz"):
        return gzip.open(bf.BlobFile(file, "rb"))
    return bf.BlobFile(file, "r")


def _read_jsonl(file: str) -> List[Dict]:
    assert bf.exists(file), file
    with open_jsonl(file) as f:
        return [json_loads(l) for l in f.readlines() if l]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,
                        default='prm')  # one of ['orm', 'prm']
    parser.add_argument(
        "--input",
        type=str,
        default=
        '/root/workspace/myoffloading/acl/myTransformer/tasks/preds/flashattn/arc-Meta-Llama-3.1-8B-Instruct'
    )
    args = parser.parse_args()
    path = args.input
    scores = dict()
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        samples_path = os.path.join(path, filename)
        samples = _read_jsonl(samples_path)
        total = 0.0
        acc = 0.0
        for sample in samples:
            pred = sample["pred"]
            answer = sample["answerKey"]
            ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
            match = re.search(ANSWER_PATTERN, pred)
            extracted_answer = match.group(1) if match else None
            total += 1.0
            acc += answer == extracted_answer if match else 0.0
        score = acc / total * 100
        print(f"{filename}: {score:.2f}")
        scores[filename] = score

    out_path = os.path.join(path, "result.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
