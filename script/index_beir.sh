#!/usr/bin/env bash

# Parameters for FAISS HNSW index
EF_CONSTRUCTION=40
EF_SEARCH=16
DIMENSION=768

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
  SAFE_MODEL_NAME=${MODEL_NAME//\//_}
  for DATASET_ID in "${DATASET_IDS[@]}"; do
    SAFE_DATASET_ID=${DATASET_ID//\//_}

    echo "Running FAISS indexing with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"

    INPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/ir_embeddings_chunks"
    OUTPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/faiss_index"

    python tool/index_embeddings.py \
      --input_dir "${INPUT_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --ef_construction "${EF_CONSTRUCTION}" \
      --ef_search "${EF_SEARCH}" \
      --dimension "${DIMENSION}"
    
    echo "---------------------------------------------------"
  done
done
