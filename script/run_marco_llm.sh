#!/usr/bin/env bash
BATCH_SIZE=100000
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

# 1. Define model-specific parameters in an associative array
declare -A GENERATE_EMBEDDINGS_PARAMS
GENERATE_EMBEDDINGS_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --doc_prompt 'passage: '"
GENERATE_EMBEDDINGS_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["nomic-ai/nomic-embed-text-v1.5"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["BAAI/bge-multilingual-gemma2"]="--normalize_embeddings"
GENERATE_EMBEDDINGS_PARAMS["BAAI/bge-m3"]="--normalize_embeddings"

declare -A QUERY_PARAMS
QUERY_PARAMS["intfloat/multilingual-e5-large"]="--normalize_embeddings --query_prompt 'query: ' --doc_prompt 'passage: '"
QUERY_PARAMS["mixedbread-ai/mxbai-embed-large-v1"]="--normalize_embeddings"
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

    PARAMS="${GENERATE_EMBEDDINGS_PARAMS[$MODEL_NAME]}"

    echo "Running embeddings generation with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"
    echo "  Parameters: ${PARAMS}"

    if [[ "$*" != *"-skip-embed"* ]]; then
      eval python tool/generate_embeddings.py \
      --model_name "${MODEL_NAME}" \
      --dataset_id "${DATASET_ID}" \
      --batch_size "${BATCH_SIZE}" \
      --flush_size 1000000 \
      --output_dir "output" \
      ${PARAMS}
    else
      echo "Skipping embeddings generation (-skip-embed flag detected)"
    fi


    echo "---------------------------------------------------"

    echo "Running FAISS indexing with:"
    echo "  Model Name: ${MODEL_NAME}"
    echo "  Dataset ID: ${DATASET_ID}"
    SAFE_MODEL_NAME=${MODEL_NAME//\//_}
    SAFE_DATASET_ID=${DATASET_ID//\//_}

    INPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/"
    OUTPUT_DIR="output/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/index/"

    if [[ "$*" != *"-skip-index"* ]]; then
      eval python tool/index_embeddings.py \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}"
    else
      echo "Skipping FAISS indexing (-skip-index flag detected)"
    fi

    echo "---------------------------------------------------"
    for QUERY_ID in "${QUERY_IDS[@]}"; do
      SAFE_QUERY_NAME=${QUERY_ID//\//_}
      
      TREC_DIR="runs/${SAFE_MODEL_NAME}/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}/"
      
      mkdir -p "${TREC_DIR}"

      TREC_FILE="${TREC_DIR}/0.trec"
      PARAMS="${QUERY_PARAMS[$MODEL_NAME]}"

      for ZERO_DIM in "${ZERO_DIMS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
          LLM_DIR="llm_docs/${SAFE_DATASET_ID}/${SAFE_QUERY_NAME}/${MODEL}"

          TREC_FILE="${TREC_DIR}/${ZERO_DIM}_${MODEL}.trec"
          echo "Running query with:"
          echo "  ZERO_DIM: ${ZERO_DIM}"
          echo "  MODEL: ${LLM_DIR}"
          
          eval python tool/query.py \
            --ir-ds-query-path "${QUERY_ID}" \
            --output-trec-name "${TREC_FILE}" \
            --index-dir "${OUTPUT_DIR}" \
            --zero-out-dims "${ZERO_DIM}" \
            --top-k 10 \
            --model ${MODEL_NAME} \
            --llm-docs ${LLM_DIR} \
            ${PARAMS}
        done
      done
    done
  done
done