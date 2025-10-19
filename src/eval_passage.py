# Date 2025.4.19
# Author nununko
# Env python 3.9


import argparse
import json
import numpy as np
import os
import pandas as pd
import re
import string

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


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def normalize_answer(s):
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


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage evaluation.")
    # In/out file params.
    parser.add_argument(
        "--input_data",
        type=str,
        default="results/pre_routed_passages/0410_tagged_asqa_23827/ckpt10423/asqa_dev_948_pre_retrieval_checkpoint_10423_retrieve_route.parquet",
        help="Path to input data.",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=5,
        help="Num of retrieved passages.",
    )
    args = parser.parse_args()

    dataset = read_file(args.input_data)

    output_dict = {"data": []}
    output_path = os.path.join(os.path.dirname(args.input_data), "passage_metrics.json")
    lens, p_scores, p_ems, p_hits = [], [], [], []
    for data in dataset:
        if "asqa" in args.input_data.lower():
            question = data["ambiguous_question"]
            qa_pairs = [{
                "context": qa_pair["context"],
                "question": qa_pair["question"],
                "short_answers": qa_pair["short_answers"].tolist(),
                "wikipage": qa_pair["wikipage"],
            } for qa_pair in data["qa_pairs"]]
            annotations = [{
                "knowledge": annotation["knowledge"].tolist(),
                "long_answer": annotation["long_answer"],
            } for annotation in data["annotations"]]
            if "router" in data.keys():
                docs = [{
                    "title": passage.split("\n")[0].strip(),
                    "text": passage,
                } for passage in data["router"]["passages"]][:args.n_docs]
            else:
                docs = data["pre_retrieved_passages"].tolist()[:args.n_docs]
        else:
            raise NotImplementedError
        
        local_lens, local_passage_scores = [0]*args.n_docs, [0]*args.n_docs
        for idx, doc in enumerate(docs):
            local_lens[idx] = compute_len([{
                "output": doc["text"],
            }])
            local_passage_scores[idx] = compute_str_em([{
                "qa_pairs": qa_pairs,
                "output": doc["text"],
            }])[0]
        if len(local_lens) > 0:
            lens.append(np.mean(local_lens))
        else:
            lens.append(0)
        p_score = 0
        if len(local_passage_scores) > 0:
            for rank, local_p_score in enumerate(local_passage_scores, 1):
                p_score += (1/rank * local_p_score)
            p_score /= len(local_passage_scores)
        p_scores.append(p_score)

        local_p_em, local_p_hit = compute_str_em([{
            "qa_pairs": qa_pairs,
            "output": "\n".join([doc["text"] for doc in docs]),
        }])
        p_ems.append(local_p_em)
        p_hits.append(local_p_hit)

        output_dict["data"].append({
            "sample_id": data["sample_id"],
            "question": question.strip(),
            "qa_pairs": qa_pairs,
            "annotations": annotations,
            "docs": docs,
            "score": {
                "len": local_lens,
                "passage_scores": local_passage_scores,
                "passage_em": local_p_em,
                "passage_hit": local_p_hit,
            },
        })

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False)
    
    print(f"len: {np.mean(lens):.2f}\np_score: {np.mean(p_scores):.2f}%")
    print(f"p_em: {np.mean(p_ems):.2f}%\np_hit: {np.mean(p_hits):.2f}%")

