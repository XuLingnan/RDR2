export PYTHONPATH=/RDR2:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python src/pipeline_inference.py \
    --dataset "datasets/asqa/asqa_train_4353.parquet" \
    --output_folder "results/asqa" \
    --retrieve \
    --online_retrieval \
    --retrieval_model "facebook/contriever-msmarco" \
    --passages "psgs_w100.tsv" \
    --passages_embeddings "wikipedia_embeddings/*" \
    --title_index "wikipedia/wikipedia_corpus/v1_cleaned_title_index.json" \
    --document_trees "wikipedia/wikipedia_corpus/wiki_pages_files/v1_cleaned_trees_jsonl" \
    --n_docs 5 \
    --route \
    --online_routing \
    --routing_model "structure-aware-rag/rdr2_llama3.1_8b_inst" \
    --prompt_template "tagged_new" \
    --read \
    --language_model "Meta-Llama-3.1-8B-Instruct" \
    --qa_format "long_form" \
    --use_tf_pipeline \
    --max_new_tokens 300 \
    --in_one_go

