#!/usr/bin/env bash
BATCH_SIZE=1000
# 1. Define an array of model names
MODEL_NAMES=(
  "BAAI/bge-multilingual-gemma2"
)

# 2. Define an array of dataset IDs
DATASET_IDS=(
  "beir/arguana"
  "beir/climate-fever"
  "beir/dbpedia-entity"
  "beir/fever"
  "beir/fiqa"
  "beir/hotpotqa"
  "beir/nfcorpus"
  "beir/nq"
  "beir/quora"
  "beir/scidocs"
  "beir/scifact"
  "beir/trec-covid"
  "beir/webis-touche2020/v2"
)

# 3. Loop over each model and dataset
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DATASET_ID in "${DATASET_IDS[@]}"; do
    
    echo "Running embeddings generation with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"
    
    python tool/generate_embeddings.py \
      --model_name "${MODEL_NAME}" \
      --dataset_id "${DATASET_ID}" \
      --batch_size "${BATCH_SIZE}" \
      --flush_size 1000000 \
      --output_dir "output" \
      --use_title
    echo "---------------------------------------------------"
  done
done
