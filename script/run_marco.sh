#!/usr/bin/env bash
BATCH_SIZE=100000
# 1. Define an array of model names
MODEL_NAMES=(
  # nomic-ai/nomic-embed-text-v1.5
  # BAAI/bge-multilingual-gemma2
  # Alibaba-NLP/gte-Qwen2-1.5B-instruct
  # BAAI/bge-m3
  # Shitao/RetroMAE_MSMARCO_distill
  # done
  # mixedbread-ai/mxbai-embed-large-v1
  # "intfloat/multilingual-e5-large"
  # "facebook/contriever-msmarco"
  "sentence-transformers/msmarco-distilbert-base-tas-b"
  # "sentence-transformers/msmarco-roberta-base-ance-firstp"
# mixedbread-ai/mxbai-embed-large-v1 # add query prompt
"Snowflake/snowflake-arctic-embed-l-v2.0" # add query prompt
    # re-done
  
  # "intfloat/multilingual-e5-large" # add query/passageprompt
  

)


# 1. Define model-specific parameters in an associative array
declare -A GENERATE_EMBEDDINGS_PARAMS
GENERATE_EMBEDDINGS_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --doc_prompt \"passage: \""
GENERATE_EMBEDDINGS_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["nomic-ai/nomic-embed-text-v1.5"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["BAAI/bge-multilingual-gemma2"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["BAAI/bge-m3"]="--normalize_embeddings"

declare -A QUERY_PARAMS
QUERY_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --query_prompt \"query: \" --doc_prompt \"passage: \""
QUERY_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings  --query_prompt \"Represent this sentence for searching relevant passages: \""
QUERY_PARAMS["Snowflake/snowflake-arctic-embed-l-v2.0"]="--query_prompt \"query: \""
QUERY_PARAMS["nomic-ai/nomic-embed-text-v1.5"]="--normalize_embeddings"
QUERY_PARAMS["BAAI/bge-multilingual-gemma2"]="--normalize_embeddings --query-prompt /home/amallia/repro-dime/bge-gemma2.prompt"
QUERY_PARAMS["BAAI/bge-m3"]="--normalize_embeddings"

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
PRF_KS=(
  # "1"
  # "2"
  # "5"
  "10"
)

# 3. Loop over each model and dataset
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DATASET_ID in "${DATASET_IDS[@]}"; do

    PARAMS="${GENERATE_EMBEDDINGS_PARAMS[$MODEL_NAME]}"

    echo "Running embeddings generation with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"
    echo "  Parameters: ${PARAMS}"

    # eval python tool/generate_embeddings.py \
    #   --model_name "${MODEL_NAME}" \
    #   --dataset_id "${DATASET_ID}" \
    #   --batch_size "${BATCH_SIZE}" \
    #   --flush_size 1000000 \
    #   --output_dir "output" \
    #   ${PARAMS}

    echo "---------------------------------------------------"

    echo "Running FAISS indexing with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"
    SAFE_MODEL_NAME=${MODEL_NAME//\//_}
    SAFE_DATASET_ID=${DATASET_ID//\//_}

    INPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/"
    OUTPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/index/"

    # eval python tool/index_embeddings.py \
    #   --input_dir "${INPUT_DIR}" \
    #   --output_dir "${OUTPUT_DIR}"
    echo "---------------------------------------------------"
    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      TREC_DIR="runs/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}/"
      
      mkdir -p "${TREC_DIR}"

      TREC_FILE="${TREC_DIR}/0.trec"
      PARAMS="${QUERY_PARAMS[$MODEL_NAME]}"

      echo "Running query with:"
      echo "  ZERO_DIM: 0"
      echo "  Parameters: ${PARAMS}"

      # eval python tool/query.py \
      #   --ir-ds-query-path "${QUERY_ID}" \
      #   --output-trec-name "${TREC_FILE}" \
      #   --index-dir "${OUTPUT_DIR}" \
      #   --zero-out-dims 0 \
      #   --top-k 10 \
      #   --model ${MODEL_NAME} \
      #   ${PARAMS}

      for ZERO_DIM in "${ZERO_DIMS[@]}"; do
        for PRF_K in "${PRF_KS[@]}"; do

          TREC_FILE="${TREC_DIR}/${ZERO_DIM}_@${PRF_K}.trec"
          echo "Running query with:"
          echo "  ZERO_DIM: ${ZERO_DIM}"
          echo "  PRF_K: ${PRF_K}"
          
          eval python tool/query.py \
            --ir-ds-query-path "${QUERY_ID}" \
            --output-trec-name "${TREC_FILE}" \
            --index-dir "${OUTPUT_DIR}" \
            --zero-out-dims "${ZERO_DIM}" \
            --top-k 10 \
            --model ${MODEL_NAME} \
            --prf-k ${PRF_K} \
            ${PARAMS}
        done
      done
    done
  done
done