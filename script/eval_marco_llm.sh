#!/usr/bin/env bash

# 1. Define an array of model names
MODEL_NAMES=(
  # M3
  BAAI/bge-m3
  # mxbai
  mixedbread-ai/mxbai-embed-large-v1
  # E5  
  "intfloat/multilingual-e5-large"
  # Snowflake
  "Snowflake/snowflake-arctic-embed-l-v2.0"
  # ANCE
  "sentence-transformers/msmarco-roberta-base-ance-firstp"
  # Contriever
  "facebook/contriever-msmarco"
  # TAS-B
  "sentence-transformers/msmarco-distilbert-base-tas-b"
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
MODELS=(
  "gpt-4.txt"
  # "gpt-4o.txt"
  # "Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4.txt"
  # "Qwen_Qwen2.5-3B-Instruct-GPTQ-Int4.txt"
  # "Qwen_Qwen2.5-7B-Instruct-GPTQ-Int4.txt"
  # "dwetzel_DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4.txt"
  # "hugging-quants_Meta-Llama-3.1-8B-Instruct-GPTQ-INT4.txt"
  # "jakiAJK_DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4.txt"
  # "meta-llama_Llama-3.2-3B-Instruct.txt"
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
      
      for MODEL in "${MODELS[@]}"; do
        for ZERO_DIM in "${ZERO_DIMS[@]}"; do

          TREC_FILE="${TREC_DIR}/${ZERO_DIM}_${MODEL}.trec"
          echo "Running eval for: ${TREC_FILE}"
          trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${TREC_FILE}"
      
        done
      done
    done
  done
done