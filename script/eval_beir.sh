#!/usr/bin/env bash

# 1. Define an array of model names
MODEL_NAMES=(
  # "sentence-transformers/msmarco-distilbert-base-tas-b"
  "Snowflake/snowflake-arctic-embed-l-v2.0"
  # "intfloat/multilingual-e5-large"
)
DATASET_IDS=(
  "beir/arguana"
  "beir/climate-fever"
  "beir/dbpedia-entity/test"
  "beir/fever/test"
  "beir/fiqa/test"
  "beir/hotpotqa/test"
  "beir/nfcorpus/test"
  "beir/nq"
  "beir/quora/test"
  "beir/scidocs"
  "beir/scifact/test"
  "beir/trec-covid"
  "beir/webis-touche2020/v2"
)

ZERO_DIMS=(
  "0.2"
  "0.4"
  "0.6"
  "0.8"
)
PRF_KS=(
  "1"
  "2"
  "5"
)

# 3. Loop over each model and dataset
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DATASET_ID in "${DATASET_IDS[@]}"; do 
    SAFE_MODEL_NAME=${MODEL_NAME//\//_}
    SAFE_DATASET_ID=${DATASET_ID//\//_}
    
    QRELS_DIR="qrels/${DATASET_ID}"
    mkdir -p ${QRELS_DIR}
    
    QRELS_FILE="${QRELS_DIR}/qrels.tsv"
    ir_datasets export ${DATASET_ID} qrels --format trec > "${QRELS_FILE}"
    
    TREC_DIR="runs/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}"
    TREC_FILE="${TREC_DIR}/0.trec"
    echo "Running eval for: ${TREC_FILE}"
    trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${TREC_FILE}"
    
    for ZERO_DIM in "${ZERO_DIMS[@]}"; do
      for PRF_K in "${PRF_KS[@]}"; do

        TREC_FILE="${TREC_DIR}/${ZERO_DIM}_@${PRF_K}.trec"
        echo "Running eval for: ${TREC_FILE}"
        trec_eval -m ndcg_cut.10 "${QRELS_FILE}" "${TREC_FILE}"      
      done
    done
  done
done