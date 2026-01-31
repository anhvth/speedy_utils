#!/usr/bin/env python3
"""
Complete working example: Process HF Dataset with Tokenization & Packing

This script demonstrates the full pattern from the skill:
1. Shard a dataset across workers
2. Each worker tokenizes and filters
3. Results merged and saved

Usage:
    python example_tokenize_pack.py \
        --src path/to/source/dataset \
        --dst path/to/output/dataset \
        --tokenizer gpt2 \
        --workers 4
"""

import os
import sys
import json
import shutil
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from datasets import load_from_disk, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from speedy_utils import multi_process
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def process_shard(args: tuple) -> Optional[str]:
    """
    Worker function: process one shard of data.
    
    Args:
        args: Tuple of (shard_id, start_idx, end_idx, src_path, 
              tokenizer_path, seq_length, temp_dir, debug)
    
    Returns:
        Path to saved temporary result, or None if shard failed/empty
    """
    shard_id, start_idx, end_idx, src_path, tokenizer_path, seq_length, temp_dir, debug = args

    # --- CRITICAL: Import inside worker ---
    import json
    import numpy as np
    from pathlib import Path
    from datasets import load_from_disk, Dataset
    from transformers import AutoTokenizer
    import shutil

    # Setup temporary paths
    shard_name = f"shard_{shard_id:05d}"
    temp_jsonl = os.path.join(temp_dir, f"{shard_name}.jsonl")
    temp_arrow = os.path.join(temp_dir, f"{shard_name}_arrow")

    try:
        # ------------------------------------------------------------------
        # STEP 1: Format & Clean (HF Dataset → Temp JSONL)
        # ------------------------------------------------------------------
        ds_local = load_from_disk(src_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        
        valid_count = 0
        
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            # Process only the assigned range for this shard
            for i in tqdm(
                range(start_idx, end_idx),
                desc=f"Shard {shard_id}",
                leave=False,
                disable=debug  # Disable progress in debug mode
            ):
                try:
                    ex = ds_local[i]
                    
                    # Example: Assume example has 'text' field
                    text = ex.get('text', '')
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    # Write valid example
                    row = {'text': text, 'original_idx': i}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    valid_count += 1
                    
                except Exception as ex_err:
                    # Skip problematic examples
                    continue
        
        if valid_count == 0:
            # Empty shard - cleanup and return None
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
            return None

        # ------------------------------------------------------------------
        # STEP 2: Tokenize (Temp JSONL → Tokenized Examples)
        # ------------------------------------------------------------------
        tokenized_examples = []
        with open(temp_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    tokens = hf_tokenizer(
                        row['text'],
                        truncation=True,
                        max_length=seq_length,
                        return_tensors=None,
                    )
                    tokenized_examples.append({
                        'input_ids': tokens['input_ids'],
                        'attention_mask': tokens.get('attention_mask', 
                                                     [1] * len(tokens['input_ids'])),
                    })
                except Exception:
                    continue
        
        if not tokenized_examples:
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
            return None

        # ------------------------------------------------------------------
        # STEP 3: Optional Packing (concatenate sequences)
        # ------------------------------------------------------------------
        # Simple packing: concatenate sequences up to max length
        packed_sequences = []
        current_packed = {'input_ids': [], 'attention_mask': []}
        
        for example in tokenized_examples:
            input_ids = example['input_ids']
            attn_mask = example['attention_mask']
            
            # If adding this would exceed limit, save current and start new
            if (len(current_packed['input_ids']) + len(input_ids)) > seq_length:
                if current_packed['input_ids']:
                    # Pad to seq_length
                    pad_len = seq_length - len(current_packed['input_ids'])
                    current_packed['input_ids'].extend([hf_tokenizer.pad_token_id] * pad_len)
                    current_packed['attention_mask'].extend([0] * pad_len)
                    packed_sequences.append(current_packed)
                current_packed = {'input_ids': [], 'attention_mask': []}
            
            # Add to current pack
            current_packed['input_ids'].extend(input_ids)
            current_packed['attention_mask'].extend(attn_mask)
        
        # Save final pack if non-empty
        if current_packed['input_ids']:
            pad_len = seq_length - len(current_packed['input_ids'])
            current_packed['input_ids'].extend([hf_tokenizer.pad_token_id] * pad_len)
            current_packed['attention_mask'].extend([0] * pad_len)
            packed_sequences.append(current_packed)
        
        if not packed_sequences:
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
            return None

        # ------------------------------------------------------------------
        # STEP 4: Save to Arrow format
        # ------------------------------------------------------------------
        ds_arrow = Dataset.from_dict({
            'input_ids': [ex['input_ids'] for ex in packed_sequences],
            'attention_mask': [ex['attention_mask'] for ex in packed_sequences],
        })
        ds_arrow.save_to_disk(temp_arrow)
        
        # Cleanup JSONL
        if os.path.exists(temp_jsonl):
            os.remove(temp_jsonl)
        
        logger.info(f"Shard {shard_id}: {len(packed_sequences)} packed sequences")
        return temp_arrow

    except Exception as e:
        logger.error(f"Shard {shard_id} failed: {str(e)[:200]}")
        # Cleanup on error
        for path in [temp_jsonl, temp_arrow]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.remove(path)
                    except:
                        pass
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process HuggingFace dataset with tokenization & packing"
    )
    parser.add_argument("--src", type=str, required=True, help="Source HF dataset path")
    parser.add_argument("--dst", type=str, required=True, help="Output HF dataset path")
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", help="Tokenizer name (HF hub or path)"
    )
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers (default: cpu_count - 1)"
    )
    parser.add_argument(
        "--backend", type=str, default="mp", choices=["mp", "ray"],
        help="multi_process backend"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: process only 1000 rows"
    )
    args = parser.parse_args()

    start_time = time.time()
    
    # Determine worker count
    if args.workers is None:
        args.workers = max(1, os.cpu_count() - 1)
    
    # Setup temp directory (absolute path for multiprocess workers)
    temp_dir = (Path(args.dst).parent / f".tmp_{Path(args.dst).stem}").absolute()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert to absolute paths (required for multiprocess workers)
        src_path = Path(args.src).absolute()
        dst_path = Path(args.dst).absolute()
        tokenizer_path = Path(args.tokenizer).absolute() if os.path.exists(args.tokenizer) else args.tokenizer
        
        # Load source dataset
        logger.info(f"Loading dataset from {src_path}")
        ds = load_from_disk(str(src_path))
        
        # Debug mode: truncate to smaller dataset
        if args.debug:
            debug_size = min(1000, len(ds))
            ds = ds.select(range(debug_size))
            logger.info(f"Debug mode: processing only {debug_size} rows")
        
        total_rows = len(ds)
        num_shards = min(args.workers, total_rows)
        
        logger.info(f"Processing {total_rows} rows into {num_shards} shards")
        logger.info(f"Using backend: {args.backend}, workers: {args.workers}")

        # Prepare worker arguments - distribute rows evenly
        worker_args = []
        rows_per_shard = total_rows // num_shards
        
        for i in range(num_shards):
            start_idx = i * rows_per_shard
            # Last shard gets any remaining rows
            if i == num_shards - 1:
                end_idx = total_rows
            else:
                end_idx = start_idx + rows_per_shard
            
            worker_args.append((
                i,
                start_idx,
                end_idx,
                str(src_path),
                str(tokenizer_path),
                args.seq_len,
                str(temp_dir),
                args.debug
            ))

        # Dispatch to workers
        logger.info("Dispatching to workers...")
        results = multi_process(
            process_shard,
            worker_args,
            workers=args.workers,
            backend=args.backend,
            desc="Processing Shards",
        )
        
        # Filter out None results (failed/empty shards)
        shard_paths = [r for r in results if r is not None]
        logger.info(f"Successfully processed {len(shard_paths)} shards")

        # Merge datasets
        if shard_paths:
            logger.info("Merging results...")
            datasets = [Dataset.load_from_disk(p) for p in shard_paths]
            full_ds = concatenate_datasets(datasets)
            
            # Save final dataset
            logger.info(f"Saving to {dst_path}")
            full_ds.save_to_disk(str(dst_path))
            
            logger.info(
                f"✅ Success! {len(full_ds)} packed sequences saved to {dst_path}"
            )
        else:
            logger.error("❌ No shards were produced (all failed or empty)")
        
        logger.info(f"Total time: {time.time() - start_time:.2f}s")
    
    finally:
        # Ensure cleanup happens no matter what
        if temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
