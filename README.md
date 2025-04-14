<p align="center">
    <img width="300px" src="logo.jpg" />
    <h1 align="center">Unveiling DIME</h1>
</p>

## Reproducibility, Generalizability, and Formal Analysis of Dimension Importance Estimation for Dense Retrieval

This repository contains the source code used for the experiments presented in the paper "Unveiling DIME: Reproducibility, Generalizability, and Formal Analysis of Dimension Importance Estimation for Dense Retrieval" by Cesare Campagnano, Antonio Mallia, and Fabrizio Silvetri, published at SIGIR, 2025 - [PDF](./unveiling-dime.pdf).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Available Scripts](#available-scripts)
  - [Running Experiments](#running-experiments)
  - [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Overview

DIME (Dimension Importance Estimation) is a novel approach for analyzing and understanding the importance of different dimensions in dense retrieval models. This repository provides implementations and tools for:

- Estimating dimension importance in dense retrieval models
- Analyzing the generalizability of importance estimates
- Conducting formal analysis of dimension relationships
- Reproducing experimental results from our paper

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/unveiling-dime.git
cd unveiling-dime
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Available Scripts

The repository contains several bash scripts in the `script` folder for running experiments and evaluations:

#### Run Scripts (Training and Inference)
- `run_marco.sh`: Run experiments on MS MARCO passage dataset
- `run_marco_llm.sh`: Run experiments with LLM variants on MS MARCO
- `run_marco_rerank.sh`: Run experiments with reranking on MS MARCO
- `run_marco_softmax.sh`: Run experiments with softmax scoring on MS MARCO
- `run_robust.sh`: Run experiments on TREC Robust dataset
- `run_robust_llm.sh`: Run experiments with LLM variants on TREC Robust
- `run_beir.sh`: Run experiments on BEIR benchmark

#### Evaluation Scripts
- `eval_marco.sh`: Evaluate results on MS MARCO
- `eval_marco_llm.sh`: Evaluate LLM results on MS MARCO
- `eval_marco_rerank.sh`: Evaluate reranking results on MS MARCO
- `eval_marco_softmax.sh`: Evaluate softmax scoring results on MS MARCO
- `eval_robust.sh`: Evaluate results on TREC Robust
- `eval_robust_llm.sh`: Evaluate LLM results on TREC Robust
- `eval_beir.sh`: Evaluate results on BEIR benchmark

### Running Experiments

The run scripts support various models and configurations. Here's how to use them:

1. **Basic Usage**
```bash
bash script/run_marco.sh
```

2. **Skip Specific Steps**
```bash
# Skip embedding generation
bash script/run_marco.sh -skip-embed

# Skip indexing
bash script/run_marco.sh -skip-index

# Skip both
bash script/run_marco.sh -skip-embed -skip-index
```

3. **Available Models**
The scripts support various models including:
- BAAI/bge-m3
- mixedbread-ai/mxbai-embed-large-v1
- intfloat/multilingual-e5-large
- Snowflake/snowflake-arctic-embed-l-v2.0
- sentence-transformers/msmarco-roberta-base-ance-firstp
- facebook/contriever-msmarco
- sentence-transformers/msmarco-distilbert-base-tas-b

4. **Experiment Parameters**
- Batch size: 100,000 (configurable)
- Zero-out dimensions: 0.2, 0.4, 0.6, 0.8
- PRF-K values: 1, 2, 5, 10
- Supported datasets: MS MARCO passage, TREC Robust, BEIR

### Evaluation

To evaluate the results:

1. **Basic Evaluation**
```bash
bash script/eval_marco.sh
```

The evaluation scripts will:
- Process all models and configurations
- Calculate NDCG@10 scores
- Generate results for different zero-out dimensions
- Evaluate PRF (Pseudo-Relevance Feedback) performance

2. **Evaluation Datasets**
- MS MARCO passage/trec-dl-2019/judged
- MS MARCO passage/trec-dl-2020/judged
- MS MARCO passage/trec-dl-hard

### Output Structure

The experiments generate the following directory structure:

```
output/
├── [model_name]/
│   └── [dataset_id]/
│       ├── embeddings/
│       └── index/
runs/
├── [model_name]/
│   └── [dataset_id]/
│       └── [query_id]/
│           ├── 0.trec
│           └── [zero_dim]_@[prf_k].trec
qrels/
└── [dataset_id]/
    └── [query_id]/
        └── qrels.tsv
```

## Citation

If you use this code or a modified version of it, please cite our paper:

```bibtex
@article{unveiling-dime,
  title={Unveiling DIME: Reproducibility, Generalizability, and Formal Analysis of Dimension Importance Estimation for Dense Retrieval},
  author={Campagnano, Cesare and Mallia, Antonio and Silvestri, Fabrizio},
  booktitle = {The 48th International ACM SIGIR Conference on Research and Development in Information Retrieval ({SIGIR})},
  publisher = {ACM},
  year={2025}
}
```
## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
1. Open an issue in this repository
2. Contact the authors via email
