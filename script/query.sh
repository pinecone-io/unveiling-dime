export API_KEY=pcsk_5UEzkA_614aBdbd9vD7qxmCVSsPiqGDih54VQFFpc2xf3jo1mWa4F5T9z5qcc3aPoaNPEZ

mkdir -p runs_smooth_clustering

python dime_repro_improved_local.py \
    --ir-ds-query-path msmarco-passage/trec-dl-2019/judged \
    --output-trec-name runs_smooth_clustering/repro_dime_improved_faiss_dl19.trec \
    --index-dir faiss_index/tas-b/index \
    --zero-out-dims 0.2 \
    --prf-k 5 \
    --weighted \
    --top-k 100 \
    --reuse-step1-retrieval \
    --metric "cosine" \
    --attention-type "linear" \
    --temperature 1.0

python dime_repro_improved_local.py \
    --ir-ds-query-path msmarco-passage/trec-dl-2020/judged \
    --output-trec-name runs_smooth_clustering/repro_dime_improved_faiss_dl20.trec \
    --index-dir faiss_index/tas-b/index \
    --zero-out-dims 0.2 \
    --prf-k 5 \
    --weighted \
    --top-k 100 \
    --reuse-step1-retrieval \
    --metric "cosine" \
    --attention-type "linear" \
    --temperature 1.0