import os
import faiss
import numpy as np
from tqdm import tqdm
import argparse
import json
import pandas as pd
import ir_datasets

from sentence_transformers import SentenceTransformer, util

def load_index_and_metadata(index_dir):
    """
    Loads the FAISS index and associated metadata (document IDs).
    :param index_dir: Directory containing the FAISS index and metadata files.
    :return: (index, doc_ids)
    """
    index_file = os.path.join(index_dir, "faiss_hnsw_index.bin")
    metadata_file = os.path.join(index_dir, "doc_ids.npy")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Index or metadata not found in {index_dir}")

    print(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)

    print(f"Loading document IDs from {metadata_file}...")
    doc_ids = np.load(metadata_file, allow_pickle=True)

    return index, doc_ids


def search_index(index, query_embedding, top_k=10):
    """
    Searches the FAISS index with a given query embedding.
    :param index: FAISS index object.
    :param query_embedding: Query embedding as a NumPy array of shape (dimension,).
    :param top_k: Number of nearest neighbors to retrieve.
    :return: (distances, indices) for the top-k results.
             distances/indices are arrays of length top_k.
    """
    # FAISS expects shape = (num_queries, dim). We have just 1 query.
    query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
    distances, indices = index.search(query_embedding, k=top_k)
    return distances[0], indices[0]


def get_faiss_matches(index, doc_ids, query_vector, top_k=1000):
    """
    Retrieve top_k matches from a FAISS index, returning a list of dicts:
    [
       {
         "id": str(doc_id),
         "score": float(score),
         "values": np.array(doc_embedding)
       },
       ...
    ]
    Note: FAISS returns distances (smaller is closer). To emulate a "score"
    that is larger-is-better, we use score = -distance.
    """
    distances, indices = search_index(index, query_vector, top_k=top_k)
    matches = []
    for dist, idx in zip(distances, indices):
        doc_id = doc_ids[idx]
        doc_vec = index.reconstruct(int(idx))  # retrieve the doc embedding
        score = float(dist)
        matches.append({"id": str(doc_id), "score": score, "values": doc_vec})
    return matches


parser = argparse.ArgumentParser(
    description="Script for PRF + dimension zero-out using a local FAISS index",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# We rename --index-name to --index-dir for clarity in FAISS usage

parser.add_argument(
    "--model", type=str, required=True, help="Model name or path"
)
parser.add_argument(
    "--index-dir", type=str, required=True, help="Directory containing the FAISS index and metadata"
)
parser.add_argument(
    "--ir-ds-query-path",
    type=str,
    required=True,
    help="Path to the dataset file on ir-datasets",
)
parser.add_argument(
    "--output-trec-name",
    type=str,
    required=True,
    help="Path to save the output TREC run"
)
parser.add_argument("--top-k", type=int, default=1000, help="Number of results to retrieve")
parser.add_argument(
    "--weighted",
    action="store_true",
    help="Use weighted average (based on scores) for centroid computation"
)
parser.add_argument(
    "--zero-out-dims",
    type=float,
    default=None,
    help="Fraction of dimensions to zero out (0.0 to 1.0)"
)
parser.add_argument(
    "--score-mass-to-retain",
    type=float,
    default=None,
    help="Fraction of score mass to retain (0.0 to 1.0)"
)
parser.add_argument(
    "--prf-k",
    type=int,
    default=False,
    help="Number of top documents to use for PRF (centroid computation)"
)

# NEW ARGUMENTS (for re-ranking / new scoring)
parser.add_argument(
    "--reuse-step1-retrieval",
    action="store_true",
    help="Reuse the retrieval from step1 and only re-rank locally with new scores"
)
parser.add_argument(
    "--metric",
    type=str,
    default="dot",
    choices=["dot", "cosine"],
    help="Metric for computing new scores when re-ranking"
)
parser.add_argument(
    "--attention-type",
    type=str,
    default="linear",
    choices=["linear", "softmax"],
    help="Type of attention weighting: linear (min-max normalized) or softmax"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for the softmax attention"
)

args = parser.parse_args()

print(f"[info] Loading FAISS index from: {args.index_dir}")
index, doc_ids = load_index_and_metadata(args.index_dir)

print(f"[info] Loading query embedding model: {args.model}")
model = SentenceTransformer(args.model)

def embed_text(text: str) -> np.ndarray:
    """Embed the given text using the loaded sentence-transformer model."""
    return model.encode(text)

def produce_trec_run_lines(query_id: str, query_matches):
    """
    Produce TREC run lines from a list of matches.
    query_matches must have: {"id": str, "score": float}
    The lines have the format: qid Q0 docid rank score tag
    """
    # Sort by descending score
    query_matches.sort(key=lambda x: -x["score"])
    lines = []
    for i, match in enumerate(query_matches):
        line = f'{query_id} Q0 {match["id"]} {i+1} {match["score"]} dense'
        lines.append(line)
    return lines

def normalize(arr, t_min, t_max):
    diff = t_max - t_min
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        # Edge case: if all values are the same, return constant
        return np.full_like(arr, (t_min + t_max) / 2)
    norm_arr = (((arr - arr_min) * diff) / (arr_max - arr_min)) + t_min
    return norm_arr

def softmax(scores, temperature=1.0):
    """
    Compute softmax over an array of scores with optional temperature.
    """
    scaled_scores = scores / temperature
    exps = np.exp(scaled_scores - np.max(scaled_scores))
    return exps / np.sum(exps)

def compute_centroid(query_matches, k=None, weighted=False, attention_type="linear", temperature=1.0):
    """
    Compute the centroid of the top-k match embeddings.
    """
    if k is not None and k > 0:
        query_matches = query_matches[:k] if len(query_matches) > k else query_matches

    embeddings = []
    scores = []
    for match in query_matches:
        embeddings.append(match["values"])
        scores.append(match["score"])

    embeddings_array = np.array(embeddings, dtype=float)

    if weighted:
        if attention_type == "linear":
            # normalize scores to [0,1]
            weights = normalize(np.array(scores), 0, 1)
        else:  # "softmax"
            weights = softmax(np.array(scores), temperature)
        centroid = np.average(embeddings_array, axis=0, weights=weights)
    else:
        centroid = np.mean(embeddings_array, axis=0)

    return centroid

def zero_out_least_important_dims_in_query(centroid, query_vector, 
                                             num_dimensions_to_retain=None, 
                                             score_mass_to_retain=None):
    """
    Zero out the least important dimensions in query_vector based on the elementwise 
    product between centroid and query_vector.

    There are two mutually exclusive modes for selecting which dimensions to keep:
    
    1. Fixed number of dimensions mode:
       If score_mass_to_retain is None, then `num_dimensions_to_retain` (a fraction, e.g. 0.8)
       indicates the fixed fraction of dimensions (based on the sorted importance) to keep.
       
    2. Mass retention mode:
       If score_mass_to_retain is provided (a float between 0 and 1), then dimensions are kept 
       until their cumulative mass (from the elementwise product) reaches at least that fraction 
       of the total mass.
    
    Parameters:
      centroid: numpy array of weights or scores
      query_vector: numpy array representing the query vector
      num_dimensions_to_retain: fraction of dimensions to retain (ignored if score_mass_to_retain is provided)
      score_mass_to_retain: fraction of the total mass to retain, or None
      
    Returns:
      modified_query_vector: a copy of query_vector with less important dimensions set to 0.
    """
    
    # Ensure that only one mode is used.
    if (score_mass_to_retain is not None) and (num_dimensions_to_retain is not None) and not (score_mass_to_retain is None and num_dimensions_to_retain is None):
        raise ValueError("Either score_mass_to_retain or num_dimensions_to_retain should be specified.")
    
    itx_vec = np.multiply(centroid, query_vector)
    dim = len(itx_vec)
    
    # If we are in fixed dimensions mode:
    if score_mass_to_retain is None:
        keep_count = int(num_dimensions_to_retain * dim)
        if keep_count >= dim:
            return query_vector.copy()
        
        # Get indices sorted by importance (ascending) and keep the top ones.
        sorted_indices = np.argsort(itx_vec)
        keep_indices = set(sorted_indices[-keep_count:])
    
    # Mass retention mode:
    else:
        # Compute total mass
        total_mass = np.sum(np.log(np.abs(itx_vec)))
        if total_mass == 0:
            return query_vector.copy()
            
        target_mass = score_mass_to_retain * total_mass
        
        # Sort indices by importance in descending order.
        sorted_indices = np.argsort(itx_vec)[::-1]
        cumulative_mass = 0.0
        keep_indices = set()
        for idx in sorted_indices:
            cumulative_mass += np.log(np.abs(itx_vec[idx]))
            keep_indices.add(idx)
            if cumulative_mass >= target_mass:
                break
        
    # Create a copy of the query vector and zero-out dimensions not in keep_indices.
    modified_query_vector = query_vector.copy()
    for i in range(dim):
        if i not in keep_indices:
            modified_query_vector[i] = 0.0
            
    return modified_query_vector

def run_queries(
    ir_dataset_queries_path: str,
    output_trec_path: str,
    top_k: int,
    weighted: bool,
    zero_out_dims: float,
    score_mass_to_retain: float,
    prf_k: int,
    reuse_step1_retrieval: bool,
    metric: str,
    attention_type: str,
    temperature: float
):
    # Load IR dataset queries
    try:
        dataset = ir_datasets.load(ir_dataset_queries_path)
    except Exception:
        print(
            f"No dataset found for {ir_dataset_queries_path} on https://ir-datasets.com/"
        )
        raise

    # Compute query embeddings
    print("Computing query vectors...")
    query_vectors_map = {}
    for q in tqdm(dataset.queries_iter(), desc="Computing query vectors"):
        query_id = q[0]
        query_text = q[1]
        query_vector = embed_text(query_text)
        query_vectors_map[query_id] = query_vector
    print("Done computing query vectors")

    # Step 1: Retrieve using original query
    print("Executing first-stage queries against FAISS...")
    step1_results_map = {}
    for q_id, q_vec in tqdm(query_vectors_map.items(), desc="Executing queries"):
        matches = get_faiss_matches(index, doc_ids, q_vec, top_k=top_k)
        step1_results_map[q_id] = matches
    print("Done executing first-stage queries.")

    if zero_out_dims == 0.0:
        print('Skipping dimension zero-out since zero_out_dims is 0.0')
        final_results_map = step1_results_map
        # Produce TREC run lines
        full_trec_run = []
        for q_id, docs in final_results_map.items():
            t_lines = produce_trec_run_lines(q_id, docs)
            full_trec_run.extend(t_lines)
    else:
        # Compute new query vectors by PRF + dimension zero-out
        print("Computing new query vectors (PRF + zero-out)...")
        new_query_vectors_map = {}
        for q_id, q_result in step1_results_map.items():
            centroid = compute_centroid(
                q_result,
                k=prf_k,
                weighted=weighted,
                attention_type=attention_type,
                temperature=temperature
            )
            orig_query_vector = query_vectors_map[q_id]
            # zero_out_dims fraction -> alpha = 1 - zero_out_dims
            alpha = 1 - zero_out_dims if zero_out_dims is not None else None
            new_query_vector = zero_out_least_important_dims_in_query(
                centroid,
                orig_query_vector,
                num_dimensions_to_retain=alpha,
                score_mass_to_retain=score_mass_to_retain
            )
            new_query_vectors_map[q_id] = new_query_vector

        # If reuse_step1_retrieval is set, re-score the same docs with the new query vector
        if reuse_step1_retrieval:
            print("Re-ranking step1 retrieval results using the new query vectors...")
            final_results_map = {}
            for q_id, docs in step1_results_map.items():
                new_qvec = new_query_vectors_map[q_id]
                final_docs = []
                for match in docs:
                    doc_vec = match["values"]
                    if metric == "dot":
                        new_score = float(np.dot(new_qvec, doc_vec))
                    else:  # "cosine"
                        dot_val = np.dot(new_qvec, doc_vec)
                        normA = np.linalg.norm(new_qvec)
                        normB = np.linalg.norm(doc_vec)
                        if normA == 0 or normB == 0:
                            new_score = 0.0
                        else:
                            new_score = float(dot_val / (normA * normB))
                    final_docs.append({"id": match["id"], "score": new_score})

                # Sort descending by new_score, then slice top_k
                final_docs.sort(key=lambda x: -x["score"])
                final_docs = final_docs[:top_k]
                final_results_map[q_id] = final_docs

            # Produce TREC run lines
            full_trec_run = []
            for q_id, docs in final_results_map.items():
                t_lines = produce_trec_run_lines(q_id, docs)
                full_trec_run.extend(t_lines)
        else:
            # Otherwise, run a second FAISS query with the updated query vector
            print("Executing second-stage queries (with new query vectors) against FAISS...")
            final_query_results_map = {}
            for q_id, q_vec in tqdm(new_query_vectors_map.items(), desc="Executing new queries"):
                matches = get_faiss_matches(index, doc_ids, q_vec, top_k=top_k)
                final_query_results_map[q_id] = matches
            print("Done executing second-stage queries.")

            # Produce TREC run lines
            full_trec_run = []
            for q_id, q_result in final_query_results_map.items():
                t_lines = produce_trec_run_lines(q_id, q_result)
                full_trec_run.extend(t_lines)

    # Write TREC file
    print(f"Writing TREC run to {output_trec_path}")
    with open(output_trec_path, "w") as file:
        for line in full_trec_run:
            file.write(f"{line}\n")
    print("Done.")


if __name__ == "__main__":
    run_queries(
        ir_dataset_queries_path=args.ir_ds_query_path,
        output_trec_path=args.output_trec_name,
        top_k=args.top_k,
        weighted=args.weighted,
        zero_out_dims=args.zero_out_dims,
        score_mass_to_retain=args.score_mass_to_retain,
        prf_k=args.prf_k,
        reuse_step1_retrieval=args.reuse_step1_retrieval,
        metric=args.metric,
        attention_type=args.attention_type,
        temperature=args.temperature
    )