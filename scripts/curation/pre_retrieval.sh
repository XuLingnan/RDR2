export PYTHONPATH=RDR2:$PYTHONPATH

python src/struct_curation.py \
    --pipeline "pre_retrieval" \
    --n_down_sample 1000 \
    --start_index 0 \
    --n_docs 5 \
    --dataset "datasets/asqa/asqa_train_4353.parquet" \
    --output_folder "curated_data/pre_retrieved_dataset"