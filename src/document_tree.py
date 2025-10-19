# Date 2025.3.12
# Author nununko
# Env python 3.9

import argparse
import glob
import json
import Levenshtein
import os
import random

from typing import Dict, List


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


class TreeNode:
    def __init__(self, nid: int, ntext: str, ntype: str, span: List[int], lighted: bool=False):
        self.id = nid
        self.text = ntext
        self.type = ntype
        self.parent = None
        self.children = []
        self.span = span
        self.lighted = lighted
    

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.children.sort(key=lambda x: x.id)


class DocumentTree:
    def __init__(
        self, raw_data: Dict, with_intro: bool=True, keep_words: int=3, lighted: bool=True
    ):
        self.nodes = {}
        self.leaves = []
        self.full_text = []
        self.with_intro = with_intro
        self.keep_words = keep_words
        self.root = self.bulid_document_tree(raw_data, lighted)


    def bulid_document_tree(self, raw_data: Dict, lighted=True):
        self.nodes = {}
        self.leaves = []
        self.full_text = raw_data["full_text"]

        for data in raw_data["extracted_nodes"]:
            node = TreeNode(data["id"], data["text"], data["type"], data["span"])
            if (data["type"] == "title") or (self.with_intro and data["relation"]["up_id"] == 0):
                node.lighted = lighted
            if data["relation"]["down_ids"] == []:
                self.leaves.append(node)
                self.leaves.sort(key=lambda x: x.id)
            self.nodes[node.id] = node

        for data in raw_data["extracted_nodes"]:
            current_node = self.nodes[data["id"]]
            up_id = data["relation"]["up_id"]
            if up_id != -1:
                parent_node = self.nodes[up_id]
                parent_node.add_child(current_node)
        
        root = None
        for node in self.nodes.values():
            if node.parent is None:
                root = node
                break
        
        return root


    def get_siblings(self, node):
        if node.parent:
            return node.parent.children
        return [node]


    def light_descendants(self, node):
        """Recursively light all content descendants of a node."""
        for child in node.children:
            if child.type == "content":
                child.lighted = True
            self.light_descendants(child)


    def light_nodes(self, nodes: List):
        """Light content nodes for given nodes."""
        for node in nodes:
            # light the content siblings
            siblings = self.get_siblings(node)
            for sibling in siblings:
                if sibling.type == "content":
                    sibling.lighted = True
            # light content ancestors
            current_node = node
            while current_node.parent is not None:
                current_node = current_node.parent
                if current_node.type == "title":
                    break
                current_node.lighted = True
            # light content siblings' content descendants
            for sibling in siblings:
                if sibling.type == "content":
                    self.light_descendants(sibling)


    def light_nodes_randomly(
        self,
        nums: List=[1, 2, 3, 4, 5],
        probs: List=[0.5, 0.2, 0.15, 0.1, 0.05],
    ):
        expand_num = random.choices(nums, probs, k=1)[0]
        expand_nodes = random.choices(self.leaves, k=expand_num)
        self.light_nodes(expand_nodes)
        return [node.id for node in expand_nodes if node.type == "content"]


    def light_nodes_from_passages(self, passages: List, metric="edit_distance"):
        expand_nodes = []
        if metric == "edit_distance":
            for passage in passages:
                boundary = len(self.full_text)-len(passage)+1
                if boundary <= 1:
                    # special condition where len(self.full_text) < len(passage)
                    expand_nodes.extend(
                        self.nodes.values()
                    )
                else:
                    chunks = [{
                        "chunk": self.full_text[i:i+len(passage)],
                        "start": i,
                        "end": i+len(passage),
                    } for i in range(boundary)]
                    # include recurrence passages conform to dpr `psgs_w100`
                    for i in range(boundary, len(self.full_text)):
                        local_chunk = self.full_text[i:] + self.full_text[:i-boundary]
                        chunks.append({
                            "chunk": local_chunk,
                            "start": i,
                            "end": i-boundary,
                        })
                    assert len(chunks) == len(self.full_text), f"{self.root.text}\npassage: {passage}\nfull_text_length-{len(self.full_text)}\npassage_length-{len(passage)}"
                    target_chunk = min(
                        chunks, key=lambda x: Levenshtein.distance(x["chunk"], passage)
                    )
                    passage_start_idx = target_chunk["start"]
                    passage_end_idx = target_chunk["end"]
                    node_start_idx, node_end_idx = -1, -1
                    for node_id in range(len(self.nodes)):
                        local_node = self.nodes[node_id]
                        if (local_node.id == 0 or local_node.type == "content"):
                        # NOTE: ignore `title` nodes when match nodes with chunks
                            if (passage_start_idx >= local_node.span[0]
                                and passage_start_idx <= local_node.span[-1]):
                                node_start_idx = node_id
                            if (passage_end_idx >= local_node.span[0]
                                and passage_end_idx <= local_node.span[-1]):
                                node_end_idx = node_id
                    assert node_start_idx != -1 and node_end_idx != -1, f"Can't match passage in DT:\n{passage}"
                    if passage_start_idx < passage_end_idx:
                        expand_nodes.extend(
                            [self.nodes[node_id] for node_id in range(node_start_idx, node_end_idx+1)]
                        )
                    else:
                        expand_nodes.extend(
                            [self.nodes[node_id] for node_id in range(node_start_idx, len(self.nodes))]
                        )
                        # NOTE: We do NOT light nodes[0:end] in the recurrence passages,
                        # as the intro nodes are always lighted and if we do need these
                        # information, we would expect the retriever to return nodes[s1=0:e1]
                        # rather than nodes[s2:-1] & nodes[0:e2].
        else:
            raise NotImplementedError
        self.light_nodes(expand_nodes)
        return [node.id for node in expand_nodes if node.type == "content"]


    def format_data(self, node, level: int, keep_words: int, with_index: bool=True):
        if node.type == "title":
            ntext = node.text
        elif node.type == "content":
            ntext = " ".join(node.text.split(" ")[:keep_words])
            if len(ntext) < len(node.text):
                ntext += "..."
        else:
            raise NotImplementedError
        
        if with_index:
            result = "{nid}: {ntext}".format_map({
                "nid": node.id,
                "ntext": ntext,
            })
        else:
            result = ntext

        return result


    def traverse(self, node, level: int=0, result: str="", completed: bool=False, with_index: bool=True):
        if node.lighted or completed:
            if with_index:
                result += "\t" * level
            result += self.format_data(
                node=node,
                level=level,
                keep_words=self.keep_words if completed else len(node.text.split(" ")),
                with_index=with_index,
            ) + "\n"
        for child in node.children:
            if child.lighted or completed:
                result = self.traverse(child, level+1, result, with_index=with_index)

        return result
    

    def resemble_passage(self, nodes: List):
        for node in nodes:
            node.lighted = True
            # light form node -> root
            curr_node = node
            while(curr_node.parent is not None):
                curr_node = curr_node.parent
                curr_node.lighted = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document Structure Tree(DST) & algorithm implemented on DST.",
    )
    # data args
    parser.add_argument(
        "--input_folder",
        type=str,
        default="curated_data/demo_dt",
        help="Path to the DST jsons."
    )
    args=parser.parse_args()

    input_files = sorted(
        [filename[:-5] for filename in os.listdir(args.input_folder) if filename.endswith(".json")]
    )
    for f_idx, filename in enumerate(input_files[3:]):
        raw_data = load_data(os.path.join(args.input_folder, f"{filename}.json"))[0]
  
        if f_idx == 0:
            passages = [
                [
                    "Their team was successful. In February 2004, Lobo became the first inductee to the CZW Hall of Fame. After this, Lobo remained inactive for the following two years.[2]\nAt Cage of Death 7, Zandig, who had turned on the fans, wanted to run CZW the way they used to. He called out Lobo, who was backstage for the event, and they joined each other along with Justice Pain and Nick Gage. They would later call themselves the Forefathers of CZW.[11]",
                ],
                [
                    "He wrestled primarily in Combat Zone Wrestling (CZW), where he won every major championship and was the first inductee into the CZW Hall of Fame in 2004.[2]\nAlong with Nick Gage, Justice Pain, and Ric Blade, Lobo was trained by John Zandig at the CZW Wrestling School and graduated first in the class.[3] He became one of the main wrestlers in CZW along with other graduates from the school. In February 1999, Lobo became the second Iron Man",
                    "* CZW World Tag Team Championship (1 time) - with T.C.K.\n* CZW Hall of Fame (2004)",
                ]
            ]
            for local_passage in passages:
                local_dt = DocumentTree(raw_data)
                print(local_dt.traverse(local_dt.root))
                local_dt.light_nodes_from_passages(
                    passages=local_passage,
                    metric="edit_distance",
                )
                print(local_dt.traverse(local_dt.root))
                print("*-"*100)

