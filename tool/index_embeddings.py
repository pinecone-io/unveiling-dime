import os
import numpy as np
import faiss
import argparse
from tqdm import tqdm

def load_chunk(input_dir, chunk_id):
    """
    Loads a single embedding chunk and its corresponding doc_id chunk.
    :param input_dir: Directory containing embedding and doc_id chunks.
    :param chunk_id: Chunk ID to load.
    :return: (embeddings, doc_ids)
    """
    emb_file = os.path.join(input_dir, f"embeddings_chunk_{chunk_id}.npy")
    doc_id_file = os.path.join(input_dir, f"doc_ids_chunk_{chunk_id}.npy")

    embeddings = np.load(emb_file)
    doc_ids = np.load(doc_id_file, allow_pickle=True)

    return embeddings, doc_ids

def get_chunk_ids(input_dir):
    """
    Retrieves all chunk IDs from the input directory.
    :param input_dir: Directory containing embedding and doc_id chunks.
    :return: List of chunk IDs.
    """
    chunk_ids = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.startswith("embeddings_chunk_") and file_name.endswith(".npy"):
            chunk_id = file_name.split("_")[-1].split(".")[0]
            chunk_ids.append(chunk_id)
    return chunk_ids

def index_embeddings(
    input_dir: str, 
    output_dir: str, 
    ef_construction: int = 40, 
    ef_search: int = 16, 
):
    """
    Indexes the embeddings from the input directory into a FAISS HNSW index.
    :param input_dir: Directory containing embedding chunks and doc_id chunks.
    :param output_dir: Directory to save the FAISS index and metadata.
    :param ef_construction: Parameter for HNSW index construction (default=40).
    :param ef_search: Parameter for HNSW search (default=16).
    :param dimension: Dimensionality of the embeddings (default=768).
    """
    os.makedirs(output_dir, exist_ok=True)


    all_doc_ids = []

    chunk_ids = get_chunk_ids(input_dir)
    print(f"[indexer] Found {len(chunk_ids)} chunks to process.")

    
    # get the dimension
    embeddings, _ = load_chunk(input_dir, chunk_ids[0])
    dimension = len(embeddings[0])
    print(f"[indexer] Embeddings dimension: {dimension}.")
    
    print("[indexer] Creating HNSW index...")
    # index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)  # 32 is the default number of neighbors
    # index.hnsw.efConstruction = ef_construction
    # index.hnsw.efSearch = ef_search
    
    index = faiss.IndexFlatIP(dimension)
    
    
    for chunk_id in tqdm(chunk_ids, desc="Indexing chunks"):
        print(f"[indexer] Processing chunk {chunk_id}...")
        embeddings, doc_ids = load_chunk(input_dir, chunk_id)

        index.add(embeddings)
        all_doc_ids.extend(doc_ids)

    index_file = os.path.join(output_dir, "faiss_hnsw_index.bin")
    metadata_file = os.path.join(output_dir, "doc_ids.npy")

    print("[indexer] Saving the FAISS index...")
    faiss.write_index(index, index_file)
    np.save(metadata_file, np.array(all_doc_ids, dtype=object))

    print(f"[indexer] FAISS index saved to {index_file}.")
    print(f"[indexer] Document IDs saved to {metadata_file}.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Index embeddings into a FAISS HNSW index."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing embedding chunks and doc_id chunks (required)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the FAISS index and metadata (required)."
    )
    parser.add_argument(
        "--ef_construction",
        type=int,
        default=40,
        help="efConstruction parameter for HNSW index (default: 40)."
    )
    parser.add_argument(
        "--ef_search",
        type=int,
        default=16,
        help="efSearch parameter for HNSW index (default: 16)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    index_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
