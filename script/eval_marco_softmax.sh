#!/usr/bin/env bash

# 1. Define an array of model names
MODEL_NAMES=(
  # "facebook/contriever-msmarco"
  # "sentence-transformers/msmarco-distilbert-base-tas-b"
  # "sentence-transformers/msmarco-roberta-base-ance-firstp"
  # "intfloat/multilingual-e5-large"
  "Snowflake/snowflake-arctic-embed-l-v2.0"
  # mixedbread-ai/mxbai-embed-large-v1
  # BAAI/bge-m3
)
DATASET_IDS=(
  "msmarco-passage"
)

QUERY_IDS=(
  "msmarco-passage/trec-dl-2019/judged"
  "msmarco-passage/trec-dl-2020/judged"
  "msmarco-passage/trec-dl-hard"
)

ZERO_DIMS=(
  "0.2"
  "0.4"
  "0.6"
  "0.8"
)
TEMPS=(
  0.1
  # 1.0
)

# 3. Loop over each model and dataset
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DATASET_ID in "${DATASET_IDS[@]}"; do 
    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_MODEL_NAME=${MODEL_NAME//\//_}
      SAFE_DATASET_ID=${DATASET_ID//\//_}
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      QRELS_DIR="qrels/${DATASET_ID}/${SAFE_QUERY_NAME}"
      mkdir -p ${QRELS_DIR}
      
      QRELS_FILE="${QRELS_DIR}/qrels.tsv"
      ir_datasets export ${QUERY_ID} qrels --format trec > "${QRELS_FILE}"
      
      TREC_DIR="runs/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}"
      
      for ZERO_DIM in "${ZERO_DIMS[@]}"; do
        for TEMP in "${TEMPS[@]}"; do

          TREC_FILE="${TREC_DIR}/${ZERO_DIM}_weighted-softmax-${TEMP}.trec"
          echo "Running eval for: ${TREC_FILE}"
          trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${TREC_FILE}"
      
        done
      done
    done
  done
done