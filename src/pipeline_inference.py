# Date 2025.4.8
# Author nununko
# Env python 3.9

from document_tree import DocumentTree
from retrieval.passage_retrieval import Retriever
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer
)
from vllm import (
    LLM, LLMEngine, SamplingParams
)

import argparse
import gc
import json
import os
import pandas as pd
import re
import string
import torch
torch.cuda.empty_cache()  # empty cache
torch.manual_seed(29)
import transformers
import unicodedata


PROMPT_DICT = {
    "router_tagged": (
        "You are asked to identify relevant nodes in a document tree that can answer the given question. "
        "Use [ANSWER] if a paragraph directly contributes to answering the question. "
        "Use [EXPAND] if a collapsed heading might contain information that can answer the question. "
        "If neither exists, reply exactly \"Cannot answer\".\n"
        "## Question\n{question}\n\n"
        "## Document\n{context}\n\n"
        "## Response"
    ),
    "router_enclosed": (
        "You are asked to identify relevant nodes in a document tree that can answer the given question. "
        "First scan visible pargraphs for content that directly contributes to answering the question, "
        "enclose such nodes within <answer></answer>. "
        "Then inspect collapsed headings for possibly hidden answers based on their structure or title, "
        "enclose such nodes within <expand></expand>. "
        "If neither exists, reply exactly \"Cannot answer\".\n"
        "## Question\n{question}\n\n"
        "## Document\n{context}\n\n"
        "## Response"
    ),
    "reader_long_form": (
        "Instruction: Write an accurate, engaging, and concise answer for the given question. Use an "
        "unbiased and journalistic tone.\n"
        "## Paragraph\n{paragraph}\n\n"
        "## Question\n{question}\n\n"
        "## Response"
    ),
    "reader_short_form": (
        "Instruction: Provide one accurate answer for the given question. "
        "Do not explain yourself or output anything else.\n"
        "## Paragraph\n{paragraph}\n\n"
        "## Question\n{question}\n\n"
        "## Response"
    ),
    "reader_list_form": (
        "Instruction: Provide a list of accurate answers for the given question. Separate answers by "
        "commas. Do not explain yourself or output anything else.\n"
        "## Paragraph\n{paragraph}\n\n"
        "## Question\n{question}\n\n"
        "## Response"
    ),
}

TASK_INST = {
    "asqa": (
        "Answer the following question. "
        "The question may be ambiguous and have multiple correct answers, "
        "and in that case, you have to provide a long-form answer including all correct answers.\n"
        "## Input:\n\n{question}"
    )
}

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


def fix_punctuation(input_string: str):
    punctuation_indices = [i for i, char in enumerate(input_string) if char in string.punctuation]

    if not punctuation_indices:
        print(f">>>Warning: Model response without punctuations\n{input_string}>>>")
        return ""
    else:
        last_punct_index = punctuation_indices[-1]
        return input_string[:last_punct_index + 1]


def parse_tagged_behavior(input_string: str):
    pattern = r'^\[([A-Za-z]+)\]\s*(\d+)\s*:\s*(.*)$'
    match = re.match(pattern, input_string)
    
    if match:
        letters = match.group(1)
        numbers = match.group(2)
        text = match.group(3)
        return letters, numbers, text
    else:
        return None, None, None


def parse_enclosed_behavior(input_string: str):
    answers = re.findall(r'<answer>(.*?)</answer>', input_string, re.DOTALL)
    expands = re.findall(r'<expand>(.*?)</expand>', input_string, re.DOTALL)
    
    return answers, expands


def check_content_descendants(node, subtree_node_ids):
    for child in node.children:
        if child.type == "content":
            if child.id in subtree_node_ids:
                return False
        # Recursively check children's descendants
        if not check_content_descendants(child, subtree_node_ids):
            return False
    return True


def route_post_process(
    pred_str, dt_ctx, idx, f_title_index, p_docutment_trees, prompt_template
):
    """
    Args:
        pred_str: router response text.
        dt_ctx: retrieval subtree.
        idx: data idx.
        f_title_index: title index file.
        p_document_trees: document trees path.
        prompt_template: router prompt template.
    Returns:
        dict{
            answer: list, answer node ids.
            expand: list, expand node ids.
            expelled: list, node ids for illegal behaviors.
            collapsed: bool, collapsed model response.
        }
    """
    # reconstruct document tree
    wiki_title = dt_ctx.strip().split("\n")[0].split(":", 1)[1].strip()
    nodes = get_nodes_from_title(wiki_title, f_title_index, p_docutment_trees)
    if nodes is None:
        print(f">>>Warning: no document found: [{idx}] {wiki_title}.>>>")
    else:
        dt = DocumentTree(nodes, with_intro=False)
    subtree_node_ids = [
        int(node_str.split(":", 1)[0].lstrip("\t"))
        for node_str in dt_ctx.strip().split("\n")
    ]

    # extract routing behavior
    answer_node_ids, expand_node_ids, expelled_node_ids = [], [], []
    refused = False
    if "cannot answer" in pred_str.lower():
        refused = True
    else:
        if "tagged" in prompt_template:
            # extract tagged format routing behavior
            behavior_strs = [
                item.strip()
                for item in pred_str.strip().split("\n")
                if item.strip() != ""
            ]
            for behavior_str in behavior_strs:
                tag, node_id, first_words = parse_tagged_behavior(behavior_str)
                if (
                    (tag != None and tag.lower() in ["expand", "answer"]) 
                    and (node_id != None and int(node_id) in subtree_node_ids) 
                    and (first_words != None)
                ):
                    if tag.lower() == "expand" and dt.nodes[int(node_id)].type == "title":
                        # expand nodes should not contain any content descendants
                        if check_content_descendants(dt.nodes[int(node_id)], set(subtree_node_ids)):
                            expand_node_ids.append(int(node_id))
                        else:
                            expelled_node_ids.append(int(node_id))
                            print(f">>>Warning: ambiguous behavior: [{idx}] {behavior_str}.>>>")
                    elif tag.lower() == "answer" and dt.nodes[int(node_id)].type == "content":
                        answer_node_ids.append(int(node_id))
                    else:
                        expelled_node_ids.append(int(node_id))
                        print(f">>>Warning: ambiguous behavior: [{idx}] {behavior_str}.>>>")
                else:
                    print(f">>>Warning: local behavior collapsed: [{idx}] {behavior_str}.>>>")
        elif prompt_template == "enclosed":
            # extract enclosed format routing behavior
            answer_strs, expand_strs = parse_enclosed_behavior(pred_str.strip())
            answer_behavior_strs = [
                item.strip()
                for answer_str in answer_strs
                for item in answer_str.strip().split("\n")
                if item.strip() != ""
            ]
            expand_behavior_strs = [
                item.strip()
                for expand_str in expand_strs
                for item in expand_str.strip().split("\n")
                if item.strip() != ""
            ]
            for answer_behavior_str in answer_behavior_strs:
                try:
                    node_id = answer_behavior_str.split(":", 1)[0]
                    first_words = answer_behavior_str.split(":", 1)[1]
                    assert int(node_id) in subtree_node_ids
                    if dt.nodes[int(node_id)].type == "content":
                        answer_node_ids.append(int(node_id))
                    else:
                        expelled_node_ids.append(int(node_id))
                        print(f">>>Warning: ambiguous behavior: [{idx}] {answer_behavior_str}.>>>")
                except:
                    print(f">>>Warning: local behavior collapsed: [{idx}] {answer_behavior_str}.>>>")
            for expand_behavior_str in expand_behavior_strs:
                try:
                    node_id = expand_behavior_str.split(":", 1)[0]
                    first_words = expand_behavior_str.split(":", 1)[1]
                    assert int(node_id) in subtree_node_ids
                    if dt.nodes[int(node_id)].type == "title":
                        # expand nodes should not contain any content descendants
                        if check_content_descendants(dt.nodes[int(node_id)], set(subtree_node_ids)):
                            expand_node_ids.append(int(node_id))
                        else:
                            expelled_node_ids.append(int(node_id))
                            print(f">>>Warning: ambiguous behavior: [{idx}] {expand_behavior_str}.>>>")
                    else:
                        expelled_node_ids.append(int(node_id))
                        print(f">>>Warning: ambiguous behavior: [{idx}] {expand_behavior_str}.>>>")
                except:
                    print(f">>>Warning: local behavior collapsed: [{idx}] {expand_behavior_str}.>>>")
        else:
            raise NotImplementedError
    
    return {
        "answer": sorted(answer_node_ids),
        "expand": sorted(expand_node_ids),
        "expelled": expelled_node_ids,
        "collapsed": (
            not refused
            and answer_node_ids == []
            and expand_node_ids == []
            and expelled_node_ids == []
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve-Route-Read Pipeline.")
    # Initialization params.
    parser.add_argument(
        "--language_model",
        type=str,
        default="Meta-Llama-3.1-8B-Instruct",
        help="Path to the language model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--routing_model",
        type=str,
        default="structure-aware-rag/rdr2_llama3.1_8b_inst",
        help="Path to the route model.",
    )
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
        "--prompt_template",
        type=str,
        choices=["tagged", "enclosed"],
        help="Different sft prompts.",
    )
    parser.add_argument(
        "--qa_format",
        type=str,
        choices=["long_form", "short_form", "list_form"],
        help="QA formats (for corresponding reader prompts). "
        "`long_form` for ASQA & ELI5; "
        "`short_form` for HotpotQA & TriviaQA; "
        "`list_form` for QAMPARI.",
    )
    parser.add_argument(
        "--with_selftext",
        action="store_true",
        help="Add `selftext`(an elaboration of the question) for the questions from eli5 dataset.",
    )
    # Retrieval params.
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="With retrieval.",
    )
    parser.add_argument(
        "--online_retrieval",
        action="store_true",
        help="Retrieve on-the-fly.",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=5,
        help="Retrieve top-[n_docs] documents.",
    )
    # Routing params.
    parser.add_argument(
        "--route",
        action="store_true",
        help="With routing.",
    )
    parser.add_argument(
        "--online_routing",
        action="store_true",
        help="Route on-the-fly.",
    )
    parser.add_argument(
        "--max_routing_iters",
        type=int,
        default=10,
        help="Max routing iiterations.",
    )
    parser.add_argument(
        "--no_expand",
        action="store_true",
        help="w/o [expand] behavior.",
    )
    parser.add_argument(
        "--no_refuse",
        action="store_true",
        help="w/o [refuse] behavior.",
    )
    parser.add_argument(
        "--no_structure",
        action="store_true",
        help="w/o structure nodes.",
    )
    parser.add_argument(
        "--no_router",
        action="store_true",
        help="Further w/o router when w/o structure is activated.",
    )
    parser.add_argument(
        "--no_similarity",
        action="store_true",
        help="w/o similarity nodes.",
    )
    parser.add_argument(
        "--no_content",
        action="store_true",
        help="Further w/o initial content node when w/o similarity is activated.",
    )
    parser.add_argument(
        "--with_intro",
        action="store_true",
        help="always send an intro subtree.",
    )
    parser.add_argument(
        "--test_time_scaling",
        action="store_true",
        help="Inference time scaling settings.",
    )
    # Reading params.
    parser.add_argument(
        "--read",
        action="store_true",
        help="Read.",
    )
    parser.add_argument(
        "--in_one_go",
        action="store_true",
        help="Read it in one go.",
    )
    # LM.generate params.
    parser.add_argument(
        "--use_tf_pipeline",
        action="store_true",
        help="Use pipeline for inference, otherwise AutoModelForCausalLM & tokenizer."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vllm for inference, otherwise transformers.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default="0.9",
        help="`gpu_memory_utilization` for vllm.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=131072,
        help="`max_model_len` for vllm.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Param `max_new_tokens` for `model.generate`.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Param `temperature` for `model.generate`.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Param `top_p` for `model.generate`.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Param `top_k` for `model.generate`.",
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
        default="curated_data/query_centric_dt/gold_wiki/query_centric_asqa_dev_948_gold_wiki_from_qa_pairs.parquet",
        help="Path to dataset."
        "ASQA: `curated_data/query_centric_dt/gold_wiki/query_centric_asqa_dev_948_gold_wiki_from_qa_pairs.parquet`, ",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Path to the output `json` file.",
    )
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    if args.no_router:
        assert args.no_structure, "`no_router` can be activated only if `no_structure` is activated."
    if args.no_content:
        assert args.no_similarity, "`no_content` can be activated only if `no_similarity` is activated."

    # Initializing the online retrieval model.
    if args.retrieve and args.online_retrieval:
        retriever = Retriever({})
        retriever.setup_retriever_demo(
            model_name_or_path=args.retrieval_model,
            passages=args.passages,
            passages_embeddings=args.passages_embeddings,
            save_or_load_index=False,
        )
    
    # Initializing the online routing model.
    if args.route:
        # Load mapping: title->nodes.
        title_index = read_file(args.title_index)
        if args.online_routing and not args.no_router and not args.test_time_scaling:
            if args.use_vllm:
                router = LLM(
                    args.routing_model,
                )
                router_sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_new_tokens,
                    skip_special_tokens=False,
                )
            else:
                if args.use_tf_pipeline:
                    router_pipeline = transformers.pipeline(
                        "text-generation",
                        model=args.routing_model,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
                else:
                    router_tokenizer = AutoTokenizer.from_pretrained(
                        args.routing_model,
                        padding_side="left",
                    )
                    router = AutoModelForCausalLM.from_pretrained(
                        args.routing_model,
                        torch_dtype=torch.bfloat16,
                    ).to("cuda")

    # Initializing the language model.
    if args.read:
        if args.use_vllm:
            reader = LLM(
                args.language_model,
            )
            reader_sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_new_tokens,
                skip_special_tokens=False,
            )
        else:
            if args.use_tf_pipeline:
                reader_pipeline = transformers.pipeline(
                    "text-generation",
                    model=args.language_model,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
            else:
                reader_tokenizer = AutoTokenizer.from_pretrained(
                    args.language_model,
                    padding_side="left",
                )
                reader = AutoModelForCausalLM.from_pretrained(
                    args.language_model,
                    torch_dtype=torch.bfloat16,
                ).to("cuda")

    # Load dataset.
    dataset = read_file(args.dataset)

    # Format output path.
    output_path = os.path.join(
        args.output_folder,
        os.path.splitext(os.path.basename(args.dataset))[0]
    )
    if args.route:
        output_path += ("_" + os.path.splitext(os.path.basename(args.routing_model))[0])
    if args.read:
        output_path += ("_" + os.path.splitext(os.path.basename(args.language_model))[0])
    if args.retrieve:
        output_path += "_retrieve"
    if args.route:
        output_path += "_route" 
    if args.read:
        output_path += "_read"
    output_path += ".parquet"
    print(f"\n***Data saving to {output_path}...***\n")

    # Pipeline.
    output_data = []
    for idx, data in tqdm(enumerate(dataset, 1)):
        # ASQA dataset supported
        # TODO: support ELI5 dataset
        # TODO: support QAMPARI dataset
        if (args.route and not args.retrieve and not args.read):
            # Router only.
            if args.online_routing:
                router_prompt = PROMPT_DICT[f"router_{args.prompt_template}"].format_map({
                    "question": data["question"].strip(),
                    "context": data["document_tree"].strip(),
                })
                if args.use_vllm:
                    router_preds = router.generate(
                        [router_prompt],
                        router_sampling_params,
                    )
                    router_pred_str = router_preds[0].outputs[0].text
                else:
                    if args.use_tf_pipeline:
                        router_messages = [
                            {"role": "user", "content": router_prompt},
                        ]
                        router_preds = router_pipeline(
                            router_messages,
                            max_new_tokens=args.max_new_tokens,
                        )
                        router_pred_str = router_preds[0]["generated_text"][-1]["content"]
                    else:
                        router_inputs = router_tokenizer(
                            router_prompt,
                            return_tensors="pt",
                        ).to("cuda")
                        router_preds = router.generate(
                            **router_inputs,
                            max_new_tokens=args.max_new_tokens,
                        )
                        router_pred_str = router_tokenizer.batch_decode(
                            router_preds,
                            skip_special_tokens=True,
                        )[0].split("## Response", 1)[1].strip()
                print(f"Routing result[{idx}] {router_pred_str}")
                data["router"] = {
                    "input": router_prompt,
                    "output": router_pred_str,
                }
            else:
                router_pred_str = data["router"]["output"]
            # Post process.
            data["router"]["behavior"] = route_post_process(
                router_pred_str,
                data["document_tree"],
                idx,
                title_index,
                args.document_trees,
                args.prompt_template,
            )
        else:
            # RAG pipeline.
            if "asqa" in args.dataset.lower():
                question = data["ambiguous_question"]
            elif "eli5" in args.dataset.lower():
                question = data["title"]
                if args.with_selftext:
                    if data["selftext"] != "[removed]":
                        question += data["selftext"]
            elif "qampari" in args.dataset.lower():
                question = data["question_text"]
            elif "hotpotqa" in args.dataset.lower():
                question = data["question"]
            elif "triviaqa" in args.dataset.lower():
                question = data["question"]
            else:
                raise NotImplementedError

            if args.debug:
                print(f"[{idx}] Question: {question.strip()}")

            # Retrieve
            if args.retrieve:
                if args.online_retrieval:
                    # retrieve on the fly.
                    passages = retriever.search_document_demo(
                        question.strip(),
                        n_docs=args.n_docs,
                    )
                else:
                    # read from file.
                    passages = data["pre_retrieved_passages"][:args.n_docs]
                passages_to_use = [passage["title"]+"\n"+passage["text"] for passage in passages]

            # Route
            if args.route:
                if args.test_time_scaling:
                    # retrieve top-n scaling
                    scaling_procedures = {}
                    for passage in passages[:args.n_docs]:
                        local_title = passage["title"]
                        if local_title not in scaling_procedures.keys():
                            local_nodes = data["router"]["procedure"][local_title]["nodes"]
                            for local_node in local_nodes["extracted_nodes"]:
                                local_node["relation"]["down_ids"] = local_node["relation"]["down_ids"].tolist()
                                local_node["span"] = local_node["span"].tolist()
                            scaling_procedures[local_title] = {
                                "nodes": local_nodes,
                                "expanded_content_ids": set(),
                                "expand_node_ids": set(),
                                "answer_node_ids": set(),
                                "retrieval_subtrees": [],
                                "routing_responses": [],
                                "routing_behaviors": [],
                                "expand_iters": -1,
                            }
                        else:
                            local_nodes = scaling_procedures[local_title]["nodes"]
                        tmp_dt = DocumentTree(local_nodes, with_intro=False)
                        _ = tmp_dt.light_nodes_from_passages(
                            passages=[passage["text"]],
                            metric="edit_distance",
                        )
                        tmp_dt_str = tmp_dt.traverse(tmp_dt.root)
                        # check initial subtrees
                        for pid, rsubtree in enumerate(
                            data["router"]["procedure"][local_title]["retrieval_subtrees"]
                        ):
                            if (
                                (
                                    rsubtree["type"] == "lighted_via_passage"
                                    and rsubtree["document_tree"].strip() == tmp_dt_str.strip()
                                    and rsubtree not in scaling_procedures[local_title]["retrieval_subtrees"]
                                ) or (
                                    args.max_routing_iters >= 0 and rsubtree["type"] == "lighted_via_intro"
                                )
                            ):
                                scaling_procedures[local_title]["retrieval_subtrees"].append(
                                    rsubtree
                                )
                                scaling_procedures[local_title]["routing_responses"].append(
                                    data["router"]["procedure"][local_title]["routing_responses"][pid]
                                )
                                rbehavior = data["router"]["procedure"][local_title]["routing_behaviors"][pid]
                                scaling_procedures[local_title]["routing_behaviors"].append(
                                    rbehavior
                                )
                                scaling_procedures[local_title]["answer_node_ids"].update(rbehavior["answer"])
                                scaling_procedures[local_title]["expand_node_ids"].update(rbehavior["expand"])
                                if rsubtree["type"] == "lighted_via_intro":
                                    scaling_procedures[local_title]["expand_iters"] += 1
                        
                        # expand iters scaling - iteratively check expand
                        to_expand_node_ids = scaling_procedures[local_title]["expand_node_ids"]
                        while (
                            len(scaling_procedures[local_title]["expand_node_ids"]) != 0
                            and scaling_procedures[local_title]["expand_iters"] < args.max_routing_iters
                        ):
                            scaling_procedures[local_title]["expand_iters"] += 1
                            to_expand_node_id = to_expand_node_ids.pop()
                            expanded_subtree = DocumentTree(local_nodes, with_intro=False)
                            to_expand_nodes = [
                                node for node in expanded_subtree.nodes[to_expand_node_id].children
                                if node.type == "content"
                            ]
                            expanded_subtree.light_nodes(to_expand_nodes)
                            expanded_subtree_str = expanded_subtree.traverse(expanded_subtree.root)
                            for pid, rsubtree in enumerate(
                                data["router"]["procedure"][local_title]["retrieval_subtrees"]
                            ):
                                if (
                                    rsubtree["type"] == "lighted_via_expand"
                                    and rsubtree["document_tree"].strip() == expanded_subtree_str.strip()
                                    and rsubtree not in scaling_procedures[local_title]["retrieval_subtrees"]
                                ):
                                    scaling_procedures[local_title]["retrieval_subtrees"].append(
                                        rsubtree
                                    )
                                    scaling_procedures[local_title]["routing_responses"].append(
                                        data["router"]["procedure"][local_title]["routing_responses"][pid]
                                    )
                                    rbehavior = data["router"]["procedure"][local_title]["routing_behaviors"][pid]
                                    scaling_procedures[local_title]["routing_behaviors"].append(
                                        rbehavior
                                    )
                                    scaling_procedures[local_title]["answer_node_ids"].update(rbehavior["answer"])
                                    scaling_procedures[local_title]["expand_node_ids"].update(rbehavior["expand"])
                                    to_expand_node_ids.update(rbehavior["expand"])
                    
                    # resemble router passages
                    for local_title in scaling_procedures.keys():
                        structure_dt = DocumentTree(
                            scaling_procedures[local_title]["nodes"], with_intro=False, lighted=False
                        )
                        structure_dt.resemble_passage([
                            structure_dt.nodes[answer_node_id]
                            for answer_node_id in scaling_procedures[local_title]["answer_node_ids"]
                            if answer_node_id in structure_dt.nodes.keys()
                        ])
                        routing_passage = structure_dt.traverse(structure_dt.root, with_index=False)
                        scaling_procedures[local_title]["passage"] = routing_passage.strip()
                    
                    scaling_router_passages = []
                    for passage in passages[:args.n_docs]:
                        routing_passage = scaling_procedures[passage["title"]]["passage"]
                        if routing_passage != "" and routing_passage not in scaling_router_passages:
                            scaling_router_passages.append(routing_passage)

                    data["router"] = {
                        "procedure": scaling_procedures,
                        "passages": scaling_router_passages,
                    }
                    output_data.append(data)
                    pd.DataFrame(output_data).to_parquet(output_path + "_tmp", index=False)
                    if args.debug and idx == 2:
                        break
                    continue

                # grouped by title and construct dt via title
                mapping_subtree_to_passage = {}
                mapping_expandid_to_passage = {}
                grouped_documents = {}
                passages_wo_router = []
                for passage in passages:
                    local_title = passage["title"]
                    mapping_expandid_to_passage[local_title] = {}
                    # initialize dt dict
                    if local_title not in grouped_documents:
                        local_nodes = get_nodes_from_title(
                            local_title, title_index, args.document_trees
                        )
                        if local_nodes is not None:
                            grouped_documents[local_title] = {
                                "nodes": local_nodes,
                                "expanded_content_ids": set(),  # avoid redundant expand behavior
                                "expand_node_ids": set(),  # used for routing iteration
                                "answer_node_ids": set(),  # used to construct routing chunk
                                "retrieval_subtrees": [],  # router inputs
                                "routing_responses": [],  # router outputs
                                "routing_behaviors": [],  # extracted router output behaviors
                            }
                        else:
                            print(f">>>Warning: no document found: {local_title}.>>>")
                            continue
                    # initialize retrieval subtree
                    if not args.no_structure:
                        local_dt = DocumentTree(local_nodes, with_intro=False)
                        if not args.no_similarity:
                            ctx = passage["text"]
                            lighted_content_ids = local_dt.light_nodes_from_passages(
                                passages=[ctx],
                                metric="edit_distance",
                            )
                        else:
                            if not args.no_content:
                                lighted_content_ids = local_dt.light_nodes_randomly(
                                    nums=[1], probs=[1.0]
                                )
                                ctx = " ".join(
                                    local_dt.nodes[lighted_id].text for lighted_id in lighted_content_ids
                                )
                            else:
                                lighted_content_ids = []
                                ctx = ""
                    else:
                        # NOTE: w/o structure, only answer or refuse, nothing to expand
                        ctx = passage["text"]
                        dt_for_structure_mapping = DocumentTree(local_nodes, with_intro=False)
                        lighted_content_ids = dt_for_structure_mapping.light_nodes_from_passages(
                            passages=[ctx],
                            metric="edit_distance",
                        )
                        local_dt = DocumentTree(local_nodes, with_intro=False, lighted=False)
                        local_dt.resemble_passage(
                            [local_dt.nodes[node_id] for node_id in lighted_content_ids]
                        )
                    local_subtree_str = local_dt.traverse(local_dt.root)
                    local_subtree_str_wo_router = local_dt.traverse(
                        local_dt.root, with_index=False
                    ).strip()
                    if local_subtree_str_wo_router not in passages_wo_router:
                        passages_wo_router.append(local_subtree_str_wo_router)
                    mapping_subtree_to_passage[local_subtree_str] = ctx
                    if local_subtree_str not in grouped_documents[local_title]["retrieval_subtrees"]:
                        grouped_documents[local_title]["retrieval_subtrees"].append({
                            "document_tree": local_subtree_str,
                            "type": "lighted_via_passage" if not args.no_similarity else "lighted_randomly",
                        })
                        grouped_documents[local_title]["expanded_content_ids"].update(
                            lighted_content_ids
                        )
                
                if args.no_structure and args.no_router:
                    data["router"] = {
                        "procedure": grouped_documents,
                        "passages": passages_wo_router,
                    }
                    passages_to_use = passages_wo_router
                    output_data.append(data)
                    pd.DataFrame(output_data).to_parquet(output_path + "_tmp", index=False)
                    if args.debug and idx == 2:
                        break
                    continue
                
                # add intro subtree if not lighted
                if args.with_intro:
                    for local_title in grouped_documents.keys():
                        local_intro_dt = DocumentTree(
                            grouped_documents[local_title]["nodes"],
                            with_intro=True,
                            lighted=(not args.no_structure),
                        )
                        local_intro_node_ids = [
                            node_id for node_id, node in local_intro_dt.nodes.items()
                            if (
                                node.parent is not None
                                and node.parent.id == 0
                                and node.type == "content"
                            )
                        ]
                        if not set(local_intro_node_ids).issubset(
                            grouped_documents[local_title]["expanded_content_ids"]
                        ):
                            # not lighted
                            if args.no_structure:
                                local_intro_dt.resemble_passage(
                                    [local_intro_dt.nodes[node_id]
                                     for node_id in local_intro_node_ids]
                                )
                            local_subtree_str = local_intro_dt.traverse(local_intro_dt.root)
                            if local_subtree_str not in grouped_documents[local_title]["retrieval_subtrees"]:
                                grouped_documents[local_title]["retrieval_subtrees"].append({
                                    "document_tree": local_subtree_str,
                                    "type": "lighted_via_intro",
                                })
                                grouped_documents[local_title]["expanded_content_ids"].update(
                                    local_intro_node_ids
                                )
                                # first paragraph in intro as recalled chunk when activating `no_refuse`
                                mapping_subtree_to_passage[local_subtree_str] = local_intro_dt.nodes[
                                    sorted(local_intro_node_ids)[0]  # TODO: check
                                ].text
                    
                num_documents = len(grouped_documents)
                # perform routing
                if args.online_routing:
                    for r_idx, local_title in enumerate(grouped_documents.keys(), 1):
                        # structure document tree for `local_title`
                        structure_dt = DocumentTree(
                            grouped_documents[local_title]["nodes"], with_intro=False, lighted=False
                        )
                        # batch prompts
                        router_prompts = [PROMPT_DICT[f"router_{args.prompt_template}"].format_map({
                            "question": question.strip(),
                            "context": subtree["document_tree"].strip(),
                        }) for subtree in grouped_documents[local_title]["retrieval_subtrees"]]
                        # batch decode
                        if args.use_vllm:
                            router_preds = router.generate(
                                router_prompts,
                                router_sampling_params,
                            )
                            router_pred_strs = [router_pred.outputs[0].text for router_pred in router_preds]
                        else:
                            if args.use_tf_pipeline:
                                batched_router_messages = [
                                    [
                                        {"role": "user", "content": router_prompt},
                                    ]
                                    for router_prompt in router_prompts
                                ]
                                batched_router_preds = router_pipeline(
                                    batched_router_messages,
                                    max_new_tokens=args.max_new_tokens,
                                )
                                router_pred_strs = [router_pred[0]["generated_text"][-1]["content"] for router_pred in batched_router_preds]
                            else:
                                router_inputs = router_tokenizer(
                                    router_prompts,
                                    padding=True,
                                    return_tensors="pt",
                                ).to("cuda")
                                router_preds = router.generate(
                                    **router_inputs,
                                    max_new_tokens=args.max_new_tokens,
                                )
                                router_pred_strs = [
                                    router_pred.split("## Response", 1)[1].strip()
                                    for router_pred in router_tokenizer.batch_decode(
                                        router_preds,
                                        skip_special_tokens=True,
                                    )
                                ]
                        grouped_documents[local_title]["routing_responses"].extend(router_pred_strs)
                        assert(
                            len(grouped_documents[local_title]["retrieval_subtrees"])
                            == len(grouped_documents[local_title]["routing_responses"])
                        )
                        if args.debug:
                            print(f"Initial routing:")
                            for init_r_idx, (r_subtree, r_response) in enumerate(zip(
                                grouped_documents[local_title]["retrieval_subtrees"],
                                grouped_documents[local_title]["routing_responses"],
                            ), 1):
                                print(f"Routing subtree[{init_r_idx}]\n{r_subtree['document_tree']}\nResponse: {r_response}\n")
                        # post_process
                        router_behaviors = [
                            route_post_process(
                                routing_response,
                                retrieval_subtree["document_tree"],
                                idx,
                                title_index,
                                args.document_trees,
                                args.prompt_template,
                            )
                            for routing_response, retrieval_subtree in zip(
                                grouped_documents[local_title]["routing_responses"],
                                grouped_documents[local_title]["retrieval_subtrees"],
                            )
                        ]
                        # answer gathering & expand iteration
                        to_expand_node_ids = set()
                        for router_behavior, retrieval_subtree in zip(
                            router_behaviors, grouped_documents[local_title]["retrieval_subtrees"]
                        ):
                            if args.no_refuse:
                                if (
                                    len(router_behavior["answer"]) == 0
                                    and (len(router_behavior["expand"]) == 0 or args.no_expand)
                                ):
                                    # reconstruct answer nodes
                                    dt_for_no_refuse = DocumentTree(
                                        grouped_documents[local_title]["nodes"], with_intro=False
                                    )
                                    recalled_node_ids = dt_for_no_refuse.light_nodes_from_passages(
                                        passages=[mapping_subtree_to_passage[
                                            retrieval_subtree["document_tree"]
                                        ]],
                                        metric="edit_distance",
                                    )
                                    if args.debug:
                                        print(f"doc {retrieval_subtree['document_tree']}")
                                        print(f"psg {mapping_subtree_to_passage[retrieval_subtree['document_tree']]}")
                                        print(f"ids {recalled_node_ids}")
                                    router_behavior["answer"] = recalled_node_ids
                            grouped_documents[local_title]["answer_node_ids"].update(
                                router_behavior["answer"]
                            )
                            for to_expand_node_id in router_behavior["expand"]:
                                # check if its content already be viewed by previous subtrees
                                if to_expand_node_id in structure_dt.nodes.keys() and check_content_descendants(
                                    structure_dt.nodes[to_expand_node_id],
                                    grouped_documents[local_title]["expanded_content_ids"],
                                ):
                                    to_expand_node_ids.add(to_expand_node_id)
                                    mapping_expandid_to_passage[local_title][to_expand_node_id] = mapping_subtree_to_passage[
                                        retrieval_subtree["document_tree"]
                                    ]
                        grouped_documents[local_title]["routing_behaviors"].extend(router_behaviors)
                        grouped_documents[local_title]["expand_node_ids"].update(to_expand_node_ids)
                        # Expand iteration
                        if not args.no_expand:
                            expand_iters = 0
                            while len(to_expand_node_ids) != 0 and expand_iters < args.max_routing_iters:
                                expand_iters += 1
                                to_expand_node_id = to_expand_node_ids.pop()
                                expanded_subtree = DocumentTree(
                                    grouped_documents[local_title]["nodes"], with_intro=False
                                )
                                to_expand_nodes = [
                                    node for node in expanded_subtree.nodes[to_expand_node_id].children
                                    if node.type == "content"
                                ]
                                if len(to_expand_nodes) == 0:
                                    continue  # nothing to expand (usually expand a nested title node whose children are title without content)
                                expanded_subtree.light_nodes(to_expand_nodes)
                                expanded_subtree_str = expanded_subtree.traverse(expanded_subtree.root)
                                if expanded_subtree_str not in grouped_documents[local_title]["retrieval_subtrees"]:
                                    grouped_documents[local_title]["retrieval_subtrees"].append({
                                        "document_tree": expanded_subtree_str,
                                        "type": "lighted_via_expand",
                                    })
                                    grouped_documents[local_title]["expanded_content_ids"].update(
                                        [node.id for node in to_expand_nodes]
                                    )
                                    router_prompt = PROMPT_DICT[f"router_{args.prompt_template}"].format_map({
                                        "question": question.strip(),
                                        "context": expanded_subtree_str.strip(),
                                    })
                                    if args.use_vllm:
                                        router_preds = router.generate(
                                            [router_prompt],
                                            router_sampling_params,
                                        )
                                        router_pred_str = router_preds[0].outputs[0].text
                                    else:
                                        if args.use_tf_pipeline:
                                            router_messages = [
                                                {"role": "user", "content": router_prompt},
                                            ]
                                            router_preds = router_pipeline(
                                                router_messages,
                                                max_new_tokens=args.max_new_tokens,
                                            )
                                            router_pred_str = router_preds[0]["generated_text"][-1]["content"]
                                        else:
                                            router_inputs = router_tokenizer(
                                                router_prompt,
                                                return_tensors="pt",
                                            ).to("cuda")
                                            router_preds = router.generate(
                                                **router_inputs,
                                                max_new_tokens=args.max_new_tokens,
                                            )
                                            router_pred_str = router_tokenizer.batch_decode(
                                                router_preds,
                                                skip_special_tokens=True,
                                            )[0].split("## Response", 1)[1].strip()
                                    if args.debug:
                                        print(f"Expanding subtree[{expand_iters}]\n{expanded_subtree_str}\nResponse: {router_pred_str}\n")
                                    grouped_documents[local_title]["routing_responses"].append(router_pred_str)
                                    # post_process
                                    router_behavior = route_post_process(
                                        router_pred_str,
                                        expanded_subtree_str,
                                        idx,
                                        title_index,
                                        args.document_trees,
                                        args.prompt_template,
                                    )
                                    # answer & expand gathering
                                    if args.no_refuse:
                                        if (
                                            len(router_behavior["answer"]) == 0 and len(router_behavior["expand"]) == 0
                                        ):
                                            recalled_node_ids = [node.id for node in to_expand_nodes]
                                            if args.debug:
                                                print(f"doc {expanded_subtree_str}")
                                                print(f"psg {mapping_subtree_to_passage[expanded_subtree_str]}")
                                                print(f"ids {recalled_node_ids}")
                                            router_behavior["answer"] = recalled_node_ids

                                    grouped_documents[local_title]["answer_node_ids"].update(
                                        router_behavior["answer"]
                                    )

                                    for local_to_expand_node_id in router_behavior["expand"]:
                                        # check if its content already be viewed by previous subtrees
                                        if local_to_expand_node_id in structure_dt.nodes.keys() and check_content_descendants(
                                            structure_dt.nodes[local_to_expand_node_id],
                                            grouped_documents[local_title]["expanded_content_ids"],
                                        ):
                                            to_expand_node_ids.add(local_to_expand_node_id)
                                            grouped_documents[local_title]["expand_node_ids"].add(
                                                local_to_expand_node_id
                                            )
                                            mapping_expandid_to_passage[local_title][
                                                local_to_expand_node_id
                                            ] = mapping_expandid_to_passage[local_title][
                                                to_expand_node_id
                                            ]
                                    grouped_documents[local_title]["routing_behaviors"].append(router_behavior)
                        # resemble routing passage
                        structure_dt.resemble_passage([
                            structure_dt.nodes[answer_node_id]
                            for answer_node_id in grouped_documents[local_title]["answer_node_ids"]
                            if answer_node_id in structure_dt.nodes.keys()
                        ])
                        routing_passage = structure_dt.traverse(structure_dt.root, with_index=False)
                        grouped_documents[local_title]["passage"] = routing_passage.strip()
                        if args.debug:
                            print(f"Document[{r_idx}]'s Routing Passage\n{routing_passage}")

                    router_passages = []
                    for passage in passages:
                        # refer to title order in passages
                        routing_passage = grouped_documents[passage["title"]]["passage"]
                        if routing_passage != "" and routing_passage not in router_passages:
                            router_passages.append(routing_passage)

                    # construct routing passages
                    data["router"] = {
                        "procedure": grouped_documents,
                        "passages": router_passages,
                    }
                    passages_to_use = router_passages
                else:
                    passages_to_use = data["router"]["passages"]

            if args.read:
                if args.in_one_go:
                    if args.retrieve or args.route:
                        reader_prompt = PROMPT_DICT[f"reader_{args.qa_format}"].format_map({
                            "paragraph": "\n\n".join(passages_to_use),
                            "question": question.strip(),
                        })
                    else:
                        reader_prompt = PROMPT_DICT[f"reader_{args.qa_format}"].format_map({
                            "paragraph": "",
                            "question": question.strip(),
                        }).replace("## Paragraph\n\n\n", "")
                    if args.use_vllm:
                        reader_preds = reader.generate(
                            [reader_prompt],
                            reader_sampling_params,
                        )
                        reader_pred_str = reader_preds[0].outputs[0].text.replace("\n", " ").strip()
                    else:
                        if args.use_tf_pipeline:
                            reader_messages = [
                                {"role": "user", "content": reader_prompt},
                            ]
                            reader_preds = reader_pipeline(
                                reader_messages,
                                max_new_tokens=args.max_new_tokens,
                            )
                            reader_pred_str = reader_preds[0]["generated_text"][-1]["content"]
                        else:
                            reader_inputs = reader_tokenizer(
                                reader_prompt,
                                return_tensors="pt",
                            ).to("cuda")
                            reader_preds = reader.generate(
                                **reader_inputs,
                                max_new_tokens=args.max_new_tokens,
                            )
                            reader_pred_str = reader_tokenizer.batch_decode(
                                reader_preds,
                                skip_special_tokens=True,
                            )[0].split("## Response", 1)[1].replace("\n", " ").strip()
                    reader_prompts = [reader_prompt]
                    reader_pred_strs = [reader_pred_str]
                else:
                    reader_prompts = [PROMPT_DICT[f"reader_{args.qa_format}"].format_map({
                        "paragraph": passage_to_use.strip(),
                        "question": question.strip(),
                    }) for passage_to_use in passages_to_use if passage_to_use.strip() != ""]
                    if args.use_vllm:
                        reader_preds = reader.generate(
                            reader_prompts,
                            reader_sampling_params,
                        )
                        reader_pred_strs = [reader_pred.outputs[0].text.replace("\n", " ").strip() for reader_pred in reader_preds]
                    else:
                        if args.use_tf_pipeline:
                            batched_reader_messages = [
                                [
                                    {"role": "user", "content": reader_prompt},
                                ]
                                for reader_prompt in reader_prompts
                            ]
                            batched_reader_preds = reader_pipeline(
                                batched_reader_messages,
                                max_new_tokens=args.max_new_tokens,
                            )
                            reader_pred_strs = [reader_pred[0]["generated_text"][-1]["content"] for reader_pred in batched_reader_preds]
                        else:
                            reader_inputs = reader_tokenizer(
                                reader_prompts,
                                padding=True,
                                return_tensors="pt",
                            ).to("cuda")
                            reader_preds = reader.generate(
                                **reader_inputs,
                                max_new_tokens=args.max_new_tokens,
                            )
                            reader_pred_strs = [
                                reader_pred.split("## Response", 1)[1].replace("\n", " ").strip()
                                for reader_pred in reader_tokenizer.batch_decode(
                                    reader_preds,
                                    skip_special_tokens=True,
                                )
                            ]
                        reader_pred_str = " ".join(
                            [fix_punctuation(pred_sent) for pred_sent in reader_pred_strs[:-1]]
                        ) + reader_pred_strs[-1]
                data["reader"] = {
                    "prompts": reader_prompts,
                    "procedure": reader_pred_strs,
                    "answer": reader_pred_str,
                }
                if args.debug:
                    print(reader_prompts)
                print(f"[{idx}]\nQ: {question.strip()}\nA: {reader_pred_str}")

        output_data.append(data)
        pd.DataFrame(output_data).to_parquet(output_path + "_tmp", index=False)
    
        if args.debug and idx == 12:
            break
        
    pd.DataFrame(output_data).to_parquet(output_path, index=False)

    print(f"\n***Data saved: {len(output_data)}.***\n")

    # Delete the llm object and free the memory
    if args.use_vllm:
        # destroy_model_parallel()
        if args.online_routing or args.pipeline == "routing":
            # del router.llm_engine.driver_worker
            del router
        if args.pipeline in [
            "retrieve-route-read",
            "retrieve-and-read",
            "no_retrieval",
        ]:
            # del reader.llm_engine.driver_worker
            del reader
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        print("Successfully delete the llm pipeline and free the GPU memory!")

