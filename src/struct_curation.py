# Date 2025.3.17
# Author nununko
# Env python 3.9

from document_tree import DocumentTree
from retrieval.passage_retrieval import Retriever
from tqdm import tqdm
from typing import Callable, List

import argparse
import json
import numpy as np
import pandas as pd
import os
import random
random.seed(29)
import re
import string
import torch
torch.manual_seed(29)
import unicodedata


def get_nodes_from_title(title, title_index, path_to_document_trees):
    title = unicodedata.normalize("NFC", title)  # normalize title
    if title in title_index:
        jsonl_filename = title_index[title]["filename"]
        line_num = title_index[title]["line_num"]
        jsonl_filepath = os.path.join(path_to_document_trees, jsonl_filename)
        
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            for current_line_num, line in enumerate(f):
                if current_line_num == line_num:
                    return json.loads(line.strip())
    else:
        return None


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval Subtree Curation.")
    # Initialization params.
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="facebook/contriever-msmarco",
        help="Path to the retrieval model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--passages",
        type=str,
        default="psgs_w100.tsv",
        help="Path to passages (.tsv file). "
        "`enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl` for demo & "
        "`psgs_w100.tsv` for entire wikipedia passages set.",
    )
    parser.add_argument(
        "--passages_embeddings",
        type=str,
        default="wikipedia_embeddings/*",
        help="Glob path to encoded passages. "
        "`enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/*` for demo & "
        "`wikipedia_embeddings/*` for entire wikipedia passages set.",
    )
    parser.add_argument(
        "--title_index",
        type=str,
        default="wikipedia/wikipedia_corpus/v1_cleaned_title_index.json",
        help="Path to title index `.json` file.",
    )
    parser.add_argument(
        "--document_trees",
        type=str,
        default="wikipedia/wikipedia_corpus/wiki_pages_files/v1_cleaned_trees_jsonl",
        help="Path to folder of document_trees constructed from wiki dump.",
    )
    # Settings.
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["document_centric", "query_centric", "pre_retrieval"],
        default="document_centric",
        help="Different pipeline settings.",
    )
    parser.add_argument(
        "--online_retrieval",
        action="store_true",
        help="Retrieve on-the-fly.",
    )
    parser.add_argument(
        "--with_selftext",
        action="store_true",
        help="Add `selftext`(an elaboration of the question) for the questions from eli5 dataset.",
    )
    parser.add_argument(
        "--group_passages",
        action="store_true",
        help="For each query, merge nodes lighted by different passages from the same article "
        "into one document subtree.",
    )
    parser.add_argument(
        "--gold_wiki",
        action="store_true",
        help="Construct document subtrees via gold wikipages.",
    )
    parser.add_argument(
        "--gold_wiki_behavior",
        type=str,
        choices=["answer", "expand", "refuse"],
        help="Construct document subtrees via gold wikipages for different routing behaviors. "
        "`answer`: light answer siblings, annotated nodes as `answer` response. "
        "`expand`: hide annotated nodes, direct parent as `expand` response, "
        "randomly light some other nodes, check with short answers. "
        "`refuse`: no short answers cases from top-5 retrieved documents.",
    )
    parser.add_argument(
        "--n_subtrees",
        type=int,
        default=3,
        help="Under `document_centric` setting, generate [1, n] subtrees per document.",
    )
    # Retrieval params.
    parser.add_argument(
        "--n_docs",
        type=int,
        default=5,
        help="Retrieve top-[n_docs] documents.",
    )
    parser.add_argument(
        "--n_down_sample",
        type=int,
        default=25000,
        help="If smaller than the dataset size, down sample to this num.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Process dataset started from this index.",
    )
    parser.add_argument(
        "--do_shuffle",
        action="store_true",
        help="Shuffle the dataset.",
    )
    parser.add_argument(
        "--with_intro",
        action="store_true",
        help="Always add intro to the document subtree."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )
    # In/out file params.
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/asqa/asqa_train_4353.parquet",
        help="Path to dataset."
        "ASQA: `datasets/asqa/asqa_train_4353.parquet`, "
        "ELI5: `datasets/eli5/explainlikeimfive/explainlikeimfive_train_272634.parquet`,"
        "WIKI: `wikipedia/wikipedia_corpus/sample_pages/v1_cleaned_1120_sampled_pages.jsonl`.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="curated_data/demo_data",
        help="Path to the output `json` file.",
    )
    args = parser.parse_args()

    # Initializing the online retrieval model.
    if args.online_retrieval or args.pipeline == "pre_retrieval":
        retriever = Retriever({})
        retriever.setup_retriever_demo(
            model_name_or_path=args.retrieval_model,
            passages=args.passages,
            passages_embeddings=args.passages_embeddings,
            save_or_load_index=False,
        )
    
    # Load dataset.
    dataset = read_file(args.dataset)
    original_dataset_size = len(dataset)
    if args.do_shuffle:
        random.shuffle(dataset)
    if args.start_index+args.n_down_sample < original_dataset_size:
        print(
            f">>>Warning: true down sample num {original_dataset_size-args.start_index} < `--n_down_sample` {args.n_down_sample}.>>>"
        )
    dataset = dataset[args.start_index:args.start_index+args.n_down_sample]
    if args.pipeline == "query_centric":
        title_index = read_file(args.title_index)

    # Pipeline.
    output_data = []
    output_path = os.path.join(
        args.output_folder,
        (
            os.path.splitext(os.path.basename(args.dataset))[0]
            + "_" + f"{args.start_index}_{min(args.start_index+args.n_down_sample, original_dataset_size)}"
            + "_" + args.pipeline + ".parquet"
        ),
    )
    print(f"\n***Data saving to {output_path}...***\n")
    for idx, data in tqdm(enumerate(dataset, 1)):
        if args.pipeline == "document_centric":
            dt_set = set()
            for subtree_idx in range(random.randint(1, args.n_subtrees)):
                local_dt = DocumentTree(data, with_intro=args.with_intro)
                _ = local_dt.light_nodes_randomly()
                local_sub_dt = local_dt.traverse(local_dt.root)
                if local_sub_dt in dt_set:
                    # remove repeated dts.
                    print(f">>>Warning: Skipped, repeated subtree for {local_dt.root.text.strip()}.>>>")
                else:
                    dt_set.add(local_sub_dt)
                    output_data.append({
                        "id": f"{idx}-{local_dt.root.text}-{subtree_idx}",
                        "question": "",
                        "document_tree": local_sub_dt,
                        "answer": "",
                    })
            pd.DataFrame(output_data).to_parquet(output_path + "_tmp", index=False)
        elif args.pipeline == "query_centric":
            if "asqa" in args.dataset.lower():
                q_id = data["sample_id"]
                question = data["ambiguous_question"]
                qa_pairs = [{
                    "context": qa_pair["context"],
                    "question": qa_pair["question"],
                    "short_answers": qa_pair["short_answers"].tolist(),
                    "wikipage": qa_pair["wikipage"],
                } for qa_pair in data["qa_pairs"]]
                wikipages = data["wikipages"].tolist()
                knowledges, long_answers = [], []
                for annotation in data["annotations"]:
                    knowledges.extend(annotation["knowledge"].tolist())
                    long_answers.append(annotation["long_answer"])
                answer = long_answers[0]  # just use the first long answer since they are equally annotated.
            elif "eli5" in args.dataset.lower():
                q_id = data["q_id"]
                question = data["title"]
                if args.with_selftext:
                    question += f'\n{data["selftext"]}'  # an elaboration of the question.
                a_ids = data["answers"]["a_id"]
                a_texts = data["answers"]["text"]
                a_scores = data["answers"]["score"]
                answer_idx = np.argmax(a_scores)
                answer = a_texts[answer_idx]  # use the answer with the highest votes.
            elif "qampari" in args.dataset.lower():
                q_id = data["qid"]
                question = data["question_text"]
                answer = ", ".join([
                    ans["answer_text"] for ans in data["answer_list"]
                ])
            else:
                raise NotImplementedError
            if args.gold_wiki:
                gold_wiki_dict = {}
                for qa_pair in qa_pairs:
                    wiki_title = qa_pair["wikipage"]
                    wiki_context = qa_pair["context"]
                    if wiki_title is not None and wiki_title not in gold_wiki_dict:
                        gold_wiki_dict[wiki_title] = set()
                    if wiki_context is not None and wiki_context != "No context provided":
                        gold_wiki_dict[wiki_title].add(wiki_context)
                # NOTE: Temporarily we just used wiki annotation in `qa_pairs` for two considerations:
                #   1. `wikipage` only provides wiki w/o context passages that cannot guarantee response existence.
                #   2. `knowledges` sometimes contains background info that is inappropriate for response annotations. 
                for local_idx, (wiki_title, wiki_contexts) in enumerate(gold_wiki_dict.items()):
                    nodes = get_nodes_from_title(wiki_title, title_index, args.document_trees)
                    if nodes is None:
                        print(f">>>Warning: no document found: {wiki_title}.>>>")  # checked
                    else:
                        if len(wiki_contexts) > 0:
                            dt_dict = {}
                            for local_subidx, wiki_context in enumerate(wiki_contexts):
                                local_dt = DocumentTree(nodes, with_intro=args.with_intro)
                                response_node_ids = local_dt.light_nodes_from_passages(
                                    passages=[wiki_context],
                                    metric="edit_distance",
                                )
                                local_sub_dt = local_dt.traverse(local_dt.root)
                                if local_sub_dt in dt_dict.keys():
                                    # remove repeated dts.
                                    print(f">>>Warning: Skipped, repeated subtree for {local_dt.root.text.strip()}, response {response_node_ids} added.>>>")
                                    # add passage idx to response.
                                    dt_dict[local_sub_dt]["passages"].append(wiki_context)
                                    dt_dict[local_sub_dt]["response_node_ids"].extend(response_node_ids)
                                else:
                                    dt_dict[local_sub_dt] = {
                                        "id": f"{q_id}-{local_idx}-{local_subidx}",
                                        "passages": [wiki_context],
                                        "response_node_ids": response_node_ids,
                                    }
                            for sub_dt, values in dt_dict.items():
                                output_data.append({
                                    "id": values["id"],
                                    "question": question.strip(),
                                    "document_tree": sub_dt,
                                    "answer": answer.strip(),
                                    "passages": list(set(values["passages"])),
                                    "response_node_ids": sorted(set(values["response_node_ids"])),
                                })
                        else:
                            continue  # see NOTE above
                            # If context were not provided, light `intro` nodes.
                            local_dt = DocumentTree(nodes, with_intro=True)
                            output_data.append({
                                "id": f"{q_id}-{local_idx}",
                                "question": question.strip(),
                                "document_tree": local_dt.traverse(local_dt.root),
                                "answer": answer.strip(),
                                "passages": [],
                                "response_node_ids": [],
                            })
            else:
                # Retrieve
                if args.online_retrieval:
                    # retrieve on the fly.
                    passages = retriever.search_document_demo(
                        question.strip(),
                        n_docs=args.n_docs,
                    )
                else:
                    # read from file.
                    passages = data["pre_retrieved_passages"][:args.n_docs]
                if args.group_passages:
                    # group passages by `title`.
                    grouped_passages = {}
                    for passage in passages:
                        local_title = passage["title"]
                        if local_title not in grouped_passages:
                            grouped_passages[local_title] = []
                        grouped_passages[local_title].append(passage)
                    for local_idx, (local_title, local_passages) in enumerate(
                        grouped_passages.items()
                    ):
                        ctxs = [passage["text"] for passage in local_passages]
                        # get nodes via `local_title`
                        nodes = get_nodes_from_title(local_title, title_index, args.document_trees)
                        if nodes is None:
                            print(f">>>Warning: no document found: {local_title}.>>>")  # checked
                        else:
                            local_dt = DocumentTree(nodes, with_intro=args.with_intro)
                            _ = local_dt.light_nodes_from_passages(
                                passages=ctxs,
                                metric="edit_distance",
                            )
                            output_data.append({
                                "id": f"{q_id}-{local_idx}",
                                "question": question.strip(),
                                "document_tree": local_dt.traverse(local_dt.root),
                                "answer": answer.strip(),
                                "passages": local_passages,
                            })
                else:
                    dt_set = set()
                    for local_idx, local_passage in enumerate(passages):
                        local_title = local_passage["title"]
                        # get nodes via `local_title`
                        nodes = get_nodes_from_title(local_title, title_index, args.document_trees)
                        if nodes is None:
                            print(f">>>Warning: no document found: {local_title}.>>>")  # checked
                        else:
                            local_dt = DocumentTree(nodes, with_intro=args.with_intro)
                            _ = local_dt.light_nodes_from_passages(
                                passages=[local_passage["text"]],
                                metric="edit_distance",
                            )
                            local_sub_dt = local_dt.traverse(local_dt.root)
                            if local_sub_dt in dt_set:
                                # remove repeated dts.
                                print(f">>>Warning: Skipped, repeated subtree for {local_dt.root.text.strip()}.>>>")
                            else:
                                dt_set.add(local_sub_dt)
                                output_data.append({
                                    "id": f"{q_id}-{local_idx}",
                                    "question": question.strip(),
                                    "document_tree": local_sub_dt,
                                    "answer": answer.strip(),
                                    "passages": [local_passage],
                                })
            pd.DataFrame(output_data).to_parquet(output_path + "_tmp", index=False)
        elif args.pipeline == "pre_retrieval":
            if "asqa" in args.dataset.lower():
                question = data["ambiguous_question"]
            elif "eli5" in args.dataset.lower():
                question = data["title"]
                if args.with_selftext:
                    question += f'\n{data["selftext"]}'  # an elaboration of the question.
            elif "qampari" in args.dataset.lower():
                question = data["question_text"]
            elif "hotpot" in args.dataset.lower():
                question = data["question"]
            elif "triviaqa" in args.dataset.lower():
                question = data["question"]
            else:
                raise NotImplementedError
            passages = retriever.search_document_demo(
                question.strip(),
                n_docs=args.n_docs,
            )
            data["pre_retrieved_passages"] = passages
            output_data.append(data)
            pd.DataFrame(output_data).to_parquet(output_path + "_tmp", index=False)
        else:
            raise NotImplementedError
        if args.debug and idx == 5:
            break

    pd.DataFrame(output_data).to_parquet(output_path, index=False)

    print(f"\n***Data saved: {len(output_data)}.***\n")

