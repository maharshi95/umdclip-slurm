"""
Module to generate embeddings for a dataset and save them to disk
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

def load_queries(dataset_name: str):
    dataset = load_dataset(dataset_name, split="queries", name="queries")
    return dataset

def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def load_dataset_shard(args: argparse.Namespace):
    dataset = load_queries(args.input)
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_path = os.path.expanduser(args.output)
    if not output_path.endswith(".npy"):
        output_path += ".npy"
    np.save(output_path, all_embeddings)
    logger.info(f"Wrote embeddings for shard {args.shard_id + 1} to {output_path}")

def add_arguments(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--shard-id", type=int, required=True, help="Zero-indexed shard number")
    parser.add_argument("--n-shards", type=int, default=1, required=True, help="Number of shards")
    parser.add_argument("--input", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, required=True, help="Output path for shard.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")


    return parser

if __name__ == "__main__":
    parser = add_arguments()
    main(parser.parse_args())