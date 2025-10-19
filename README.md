# RDR2
Code, [data](https://huggingface.co/datasets/structure-aware-rag/rdr2_train_data) and [model](https://huggingface.co/structure-aware-rag/rdr2_llama3.1_8b_inst) for the paper "Equipping Retrieval-Augmented Large Language Models with Document Structure Awareness" (EMNLP 2025 Findings).
## Installation
Install dependent Python libraries by running the command below.
```bash
pip install -r requirements.txt
```
## Download Data
Download preprocessed passage data used in DPR and the corresponding embedding files.
```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
```
Download the [Wikipedia dump 2018](https://archive.org/details/enwiki-20181220).
## Build Document Structure Tree
Execute the following codes to build document structure trees.
```bash
# build dsts
python -u parse_to_jsonlStruct.py --input_folder wiki_pages_files/raw_xmls --temp_folder wiki_pages_files/temp_extracted_pages --output_folder wiki_pages_files/cleaned_dst_jsonl --max_workers 8
# build title2tree index
python file_utils.py --task build_index --jsonl_dir wiki_pages_files/cleaned_dst_jsonl --output_index_file wiki_pages_files/cleaned_title_index.json
```
## Quick start
Execute the following code to perform the complete retrieve-route-read pipeline.
```bash
bash scripts/inference/rdr2_pipeline.sh
```
