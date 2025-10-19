# Date 2025.4.9
# Author nununko
# Env python 3.9

import argparse
import json
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def read_file(input_path):
    data = []
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
    elif input_path.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f.readlines]
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
        data = df.to_dict(orient="records")
    else:
        raise NotImplementedError
    
    print(f"\n***Loaded dataset size: {len(data)}.***\n")
    return data


def compute_routing_f1_score(response, gold):
    response_set = set(response)
    gold_set = set(gold)

    tp = len(response_set & gold_set)

    if len(response_set) == 0 or len(gold_set) == 0:
        return None, None, None
    
    precision = tp / len(response_set)
    recall = tp / len(gold_set)

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation.")
    # Settings.
    parser.add_argument(
        "--metrics",
        type=str,
        default="routing",
        choices=["routing", "routing_new"],
        help="Evaluation metrics.",
    )
    # In/out file params.
    parser.add_argument(
        "--dataset",
        type=str,
        default="results/router/ckpt3032/query_centric_asqa_dev_948_gold_wiki_from_qa_pairs_routing_checkpoint_3032.parquet",
        help="Path to dataset.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/router/ckpt3032",
        help="Path to the output `json` file.",
    )
    args = parser.parse_args()

    # Load dataset.
    dataset = read_file(args.dataset)

    output_data = []
    output_path = os.path.join(
        args.output_folder,
        os.path.splitext(os.path.basename(args.dataset))[0]
        + "_metrics.parquet"
    )
    print(f"\n***Data saving to {output_path}...***\n")

    precisions, recalls, f1_scores = [], [], []
    answer_scores = {
        "precision": [],
        "recall": [],
        "f1_score": [],
    }
    expand_scores = {
        "precision": [],
        "recall": [],
        "f1_score": [],
    }
    refuse_accs, expelled_nums, collapsed_rates = [], [], []
    for idx, data in tqdm(enumerate(dataset, 1)):
        data["metrics"] = {}
        if args.metrics == "routing":
            p, r, f = compute_routing_f1_score(
                response=data["router"]["behavior"]["answer"],
                gold=data["response_node_ids"],
            )
            if p is not None and r is not None and f is not None:
                data["metrics"] = {
                    "routing": {
                        "precision": p,
                        "recall": r,
                        "f1_score": f,
                    }
                }
                precisions.append(p)
                recalls.append(r)
                f1_scores.append(f)
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
        elif args.metrics == "routing_new":
            data["metrics"]["routing"] = {}
            data["metrics"]["routing"]["answer"] = None
            data["metrics"]["routing"]["expand"] = None
            data["metrics"]["routing"]["refuse"] = None

            answer = data["router"]["behavior"]["answer"]
            expand = data["router"]["behavior"]["expand"]
            expelled = data["router"]["behavior"]["expelled"]
            collapsed = data["router"]["behavior"]["collapsed"]
            gold_a = data["response_node_ids"]["answer_node_ids"]
            gold_e = data["response_node_ids"]["expand_node_ids"]
            # compute collapsed rates
            if collapsed:
                data["metrics"]["routing"]["cpllapsed"] = 1
                collapsed_rates.append(1)
            else:
                data["metrics"]["routing"]["cpllapsed"] = 0
                collapsed_rates.append(0)
            # compute expelled nums
            data["metrics"]["routing"]["expelled_num"] = len(expelled)
            expelled_nums.append(len(expelled))
            if data["tag"] in ["answer", "expand"]:
                if len(gold_a) > 0:
                    # shold have answer, otherwise 0
                    pa, ra, fa = 0, 0, 0
                    if len(answer) > 0:
                        pa, ra, fa = compute_routing_f1_score(
                            response=answer,
                            gold=gold_a,
                        )
                    data["metrics"]["routing"]["answer"] = {
                        "precision": pa,
                        "recall": ra,
                        "f1_score": fa,
                    }
                    answer_scores["precision"].append(pa)
                    answer_scores["recall"].append(ra)
                    answer_scores["f1_score"].append(fa)
                if len(gold_e) > 0:
                    # should have expand, otherwise 0
                    pe, re, fe = 0, 0, 0
                    if len(expand) > 0:
                        pe, re, fe = compute_routing_f1_score(
                            response=expand,
                            gold=gold_e,
                        )
                    data["metrics"]["routing"]["expand"] = {
                        "precision": pe,
                        "recall": re,
                        "f1_score": fe,
                    }
                    expand_scores["precision"].append(pe)
                    expand_scores["recall"].append(re)
                    expand_scores["f1_score"].append(fe)
            elif data["tag"] == "refuse":
                # compute refuse acc
                if len(answer) == 0 and len(expand) == 0:
                    data["metrics"]["routing"]["refuse"] = 1
                    refuse_accs.append(1)
                else:
                    data["metrics"]["routing"]["refuse"] = 0
                    refuse_accs.append(0)
            else:
                print(f">>>WARNING: idx: {idx} invalid data tag {data['tag']}.>>>")
        else:
            raise NotImplementedError
        output_data.append(data)
    pd.DataFrame(output_data).to_parquet(output_path, index=False)

    print(f"\n***Data saved: {len(output_data)}.***\n")

    if args.metrics == "routing":
        print(f"Average scores:\np-{np.mean(precisions)*100:.2f}%\nr-{np.mean(recalls)*100:.2f}%\nf-{np.mean(f1_scores)*100:.2f}%")
    elif args.metrics == "routing_new":
        print("Average scores:")
        print(f"Answer: p-{np.mean(answer_scores['precision'])*100:.2f}% r-{np.mean(answer_scores['recall'])*100:.2f}% f-{np.mean(answer_scores['f1_score'])*100:.2f}%")
        print(f"Expand: p-{np.mean(expand_scores['precision'])*100:.2f}% r-{np.mean(expand_scores['recall'])*100:.2f}% f-{np.mean(expand_scores['f1_score'])*100:.2f}%")
        print(f"refused_acc: {np.mean(refuse_accs)*100:.2f}%")
        print(f"collapsed_rate: {np.mean(collapsed_rates)*100:.2f}%")
        print(f"avg_expelled_nums: {np.mean(expelled_nums)*100:.2f}%")

