import re
import argparse
import os
import json


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    return parser.parse_args(args)


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


if __name__ == '__main__':
    args = parse_args()
    scores = dict()

    path = args.model
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue

        total_count = 0
        total_acc = 0

        count = {
            "easy": {
                "short": 0,
                "medium": 0,
                "long": 0
            },
            "hard": {
                "short": 0,
                "medium": 0,
                "long": 0
            },
        }
        acc = {
            "easy": {
                "short": 0,
                "medium": 0,
                "long": 0
            },
            "hard": {
                "short": 0,
                "medium": 0,
                "long": 0
            },
        }

        predictions, answers = [], []
        dataset = filename.split('.')[0]
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pred = extract_answer(data["pred"])
                answer = data["answer"]
                count[data["difficulty"]][data["length"]] += 1
                total_count += 1
                if pred is not None and answer == pred:
                    acc[data["difficulty"]][data["length"]] += 1
                    total_acc += 1

        for difficulty in count:
            for length in count[difficulty]:
                dataset = f"{difficulty}-{length}"
                scores[dataset] = acc[difficulty][length] / count[difficulty][
                    length] * 100
                print(f"{dataset}: {scores[dataset]:.2f}")
        total = total_acc / total_count * 100
        print(f"total: {total:.2f}")
        scores["total"] = total
    out_path = os.path.join(path, "result.json")

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
