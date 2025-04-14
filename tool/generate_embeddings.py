import os
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from sentence_transformers import SentenceTransformer
import ir_datasets
import argparse

class IRIterableDataset(IterableDataset):
    """
    An IterableDataset that reads documents from an IR dataset in a streaming fashion.
    Each item is a tuple: (doc_id, text), where text may include a title if requested.
    """
    def __init__(self, dataset_id: str, use_title: bool = False):
        super().__init__()
        self.ds = ir_datasets.load(dataset_id).docs_iter()
        self.use_title = use_title

    def __iter__(self):
        for doc in self.ds:
            text = doc.text
            # If --use_title was passed AND the doc has a non-empty title,
            # prepend the title to the text
            if self.use_title and hasattr(doc, "title") and doc.title:
                text = doc.title + " " + doc.text
            yield (doc.doc_id, text)


class RB04IterableDataset(IRIterableDataset):
    def __iter__(self):
        for doc in self.ds:
            text = doc.title + " " + doc.body
            yield (doc.doc_id, text)


def collate_fn(batch):
    """
    batch: list of (doc_id, text) pairs
    Returns:
      doc_ids: list of doc_ids (strings)
      texts: list of texts (strings)
    """
    doc_ids = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return doc_ids, texts

def main(
    model_name: str,
    dataset_id: str,
    batch_size: int = 16,
    flush_size: int = 1000,
    output_dir: str = "ir_embeddings_chunks",
    use_title: bool = False,
    dataset_type = None,
    normalize_embeddings=False,
    doc_prompt= None
):
    """
    :param model_name: which Sentence Transformers model to use (required).
    :param dataset_id: which IR dataset to load from ir_datasets (required).
    :param batch_size: how many samples (docs) per GPU per forward pass (default=16).
    :param flush_size: flush embeddings/doc_ids to disk after this many total docs (default=1000).
    :param output_dir: final sub-directory to store chunked embeddings (default='ir_embeddings_chunks').
    :param use_title: if True, prepend the document title (if present) to the text.
    """

    # --------------------------------------------------------------------------
    # Construct final output path: output/<safe_model_name>/<safe_dataset_id>/<output_dir>
    # Replace slashes in model_name or dataset_id with underscores to avoid nested dirs.
    # --------------------------------------------------------------------------
    safe_model_name = model_name.replace("/", "_")
    safe_dataset_id = dataset_id.replace("/", "_")

    final_output_dir = os.path.join(output_dir, safe_model_name, safe_dataset_id, )
    os.makedirs(final_output_dir, exist_ok=True)

    # 1) Build an IterableDataset for IR docs
    if dataset_type == "rb04":
        dataset = RB04IterableDataset(dataset_id=dataset_id, use_title=use_title)
    else:
        dataset = IRIterableDataset(dataset_id=dataset_id, use_title=use_title)

    # 2) Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0  # can set >0 if you want parallel reading
    )

    # 3) Load and prepare Sentence Transformers model
    model = SentenceTransformer(model_name)

    model.eval().cuda()

    pool = model.start_multi_process_pool()

    # Buffers on main process
    doc_ids_buffer = []
    embed_buffer = []
    total_docs_in_buffer = 0
    chunk_id = 0

    # 4) Inference loop
    with torch.no_grad():
        for doc_ids, texts in dataloader:
            embeddings = model.encode_multi_process(texts, pool, show_progress_bar=True, normalize_embeddings=normalize_embeddings, prompt=doc_prompt)

            doc_ids_buffer.append(doc_ids)
            embed_buffer.append(embeddings)
            total_docs_in_buffer += len(doc_ids)

            # Flush if we exceed flush_size
            if total_docs_in_buffer >= flush_size:
                # Concatenate
                all_doc_ids = sum(doc_ids_buffer, [])
                all_embeds = np.concatenate(embed_buffer, axis=0)

                # Save doc_ids and embeddings
                doc_id_file = os.path.join(final_output_dir, f"doc_ids_chunk_{chunk_id}.npy")
                emb_file = os.path.join(final_output_dir, f"embeddings_chunk_{chunk_id}.npy")

                np.save(doc_id_file, np.array(all_doc_ids, dtype=object))
                np.save(emb_file, all_embeds)
                print(f"[main process] Flushed {len(all_doc_ids)} docs to {doc_id_file}, {emb_file}")

                # Reset
                doc_ids_buffer.clear()
                embed_buffer.clear()
                total_docs_in_buffer = 0
                chunk_id += 1

    # 5) Final flush for leftover data
    all_doc_ids = sum(doc_ids_buffer, [])
    all_embeds = np.concatenate(embed_buffer, axis=0)

    doc_id_file = os.path.join(final_output_dir, f"doc_ids_chunk_{chunk_id}.npy")
    emb_file = os.path.join(final_output_dir, f"embeddings_chunk_{chunk_id}.npy")

    np.save(doc_id_file, np.array(all_doc_ids, dtype=object))
    np.save(emb_file, all_embeds)
    print(f"[main process] Final flush of {len(all_doc_ids)} docs to {doc_id_file}, {emb_file}")

    model.stop_multi_process_pool(pool)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode IR dataset docs with a Sentence Transformers model, "
                    "distributing across multiple GPUs. Optionally use doc titles."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name/path for Sentence Transformers (required)."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="IR dataset ID (e.g. msmarco-passage/train) for ir_datasets (required)."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        help="IR dataset type (e.g. rb04)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per device (default: 16)."
    )
    parser.add_argument(
        "--flush_size",
        type=int,
        default=1000,
        help="Flush to disk after encoding this many docs (default: 1000)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ir_embeddings_chunks",
        help="Final subdirectory name for chunked embeddings (default: ir_embeddings_chunks)."
    )
    parser.add_argument(
        "--use_title",
        action="store_true",
        help="If specified, prepend doc.title to doc.text (if available)."
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="If specified, normalize the embeddings."
    )
    parser.add_argument(
        "--doc_prompt",
        type=str,
        default=None,
        help="Path to file containing query promt."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        flush_size=args.flush_size,
        output_dir=args.output_dir,
        use_title=args.use_title,
        dataset_type=args.dataset_type,
        normalize_embeddings=args.normalize_embeddings,
        doc_prompt=args.doc_prompt
    )
