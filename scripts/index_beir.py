"""
Module to generate embeddings for a dataset and save them to disk.

This script processes a specified shard of a BeIR dataset, generates embeddings
using a pre-trained model, and saves the embeddings to disk.

Usage:
    python scripts/index_beir.py \
        --model "sentence-transformers/all-MiniLM-L6-v2" \
        --shard-id 0 \
        --n-shards 4 \
        --dataset "beir/fiqa" \
        --output-dir "/path/to/output/directory" \
        --batch-size 32 \
        --max-length 512

Arguments:
    --model: Path or name of the pre-trained model to use for embedding generation.
    --shard-id: Zero-indexed shard number to process.
    --n-shards: Total number of shards the dataset is split into.
    --dataset: Name of the BeIR dataset in HuggingFace format.
    --output-dir: Directory to save the generated embeddings.
    --batch-size: Number of samples to process in each batch (default: 8).
    --max-length: Maximum sequence length for tokenization (default: 512).
    --seed: Random seed for reproducibility (default: 42).
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from tqdm import trange
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from typing import Optional

from loguru import logger


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def load_dataset_shard(args: argparse.Namespace):
    dataset = load_dataset(args.dataset, split="queries", name="queries")
    shard_size = math.ceil(len(dataset) / args.n_shards)
    start = args.shard_id * shard_size
    end = min((args.shard_id + 1) * shard_size, len(dataset))
    return dataset.select(range(start, end))

def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    # Load model and tokenizer
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_length
    model.cuda()

    dataset_shard = load_dataset_shard(args)

    # Process shard and generate embeddings
    all_embeddings = []
    total_batches = int(math.ceil(len(dataset_shard) / args.batch_size))
    logger.info(f"total batches: {total_batches}")

    for i in trange(total_batches):
        batch = dataset_shard[i * args.batch_size : (i + 1) * args.batch_size]
        texts = [e.strip() for e in batch["text"]]
        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**model_inputs)
            emb = mean_pooling(outputs.last_hidden_state, model_inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Shard {args.shard_id + 1} embeddings shape: {all_embeddings.shape}")

    output_dir = os.path.expanduser(args.output_dir)
    sanitized_dataset_name = args.dataset.replace("/", "_")
    output_path = os.path.join(output_dir, sanitized_dataset_name, f"shard_{args.shard_id}.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_embeddings)
    logger.info(f"Wrote embeddings for shard {args.shard_id + 1} to {output_path}")

def add_arguments(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--shard-id", type=int, required=True, help="Zero-indexed shard number")
    parser.add_argument("--n-shards", type=int, default=1, required=True, help="Number of shards")
    parser.add_argument("--dataset", type=str, required=True, help="BeIR (HF) dataset name")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for shard.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")


    return parser

if __name__ == "__main__":
    parser = add_arguments()
    main(parser.parse_args())