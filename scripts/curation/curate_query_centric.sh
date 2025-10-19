export PYTHONPATH=RDR2:$PYTHONPATH

python src/struct_curation.py \
    --pipeline "query_centric" \
    --n_down_sample 25000 \
    --start_index 0 \
    --n_docs 5 \
    --dataset "curated_data/pre_retrieved_dataset/asqa_train_4353_pre_retrieval.parquet" \
    --output_folder "curated_data/query_centric_dt/pre_retrieved_wiki"