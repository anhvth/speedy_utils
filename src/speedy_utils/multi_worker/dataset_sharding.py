"""
Dataset sharding utilities for parallel processing with merge.

This module provides utilities for processing large HuggingFace datasets in parallel
by sharding them across workers, processing each shard independently, and then
merging the results back together.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Callable, Optional, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

__all__ = ['multi_process_dataset']


def multi_process_dataset(
    dataset: 'Dataset',  # type: ignore[name-defined]
    process_func: Callable,
    output_path: str,
    process_func_kwargs: Optional[Dict[str, Any]] = None,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,
    debug: bool = False,
    debug_size: int = 10000,
    backend: str = 'ray'
) -> str:
    """
    Process a dataset in parallel using sharding and multiprocessing.
    
    This function implements the shard-process-merge pattern for large dataset processing:
    1. Optionally shuffle and truncate dataset (for debugging)
    2. Shard dataset across workers
    3. Process each shard in parallel
    4. Merge results and save to final location
    5. Clean up temporary shard files
    
    Args:
        dataset: The input dataset to process
        process_func: Function to apply to each shard. Should take dataset as first argument
                    and output_path as second argument, plus any additional kwargs.
                    Must return the path to the processed shard.
        output_path: Base path for output (without extension or size suffix)
        process_func_kwargs: Additional keyword arguments to pass to process_func
        num_workers: Number of parallel workers (default: CPU count - 2)
        seed: Random seed for shuffling (if None, no shuffling)
        debug: If True, process only a subset of data for debugging
        debug_size: Number of examples to use in debug mode
        backend: Backend for multiprocessing ('ray' or 'process')
        
    Returns:
        str: Path to the final merged dataset
        
    Example:
        def process_shard(shard_dataset, output_path, tokenizer_path, seq_len):
            # Process the shard (tokenize, pack, etc.)
            packer = SFTDatasetPacker(tokenizer_path, seq_len)
            return packer.pack(shard_dataset, output_path)
        
        final_path = multi_process_dataset(
            dataset=dataset,
            process_func=process_shard,
            output_path="./data/processed",
            process_func_kwargs={
                'tokenizer_path': 'tokenizers/Qwen3-32B',
                'seq_len': 12288
            },
            num_workers=32,
            seed=42,
            debug=True
        )
    """
    from ..multi_worker import multi_process
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 2)
    
    # Shuffle dataset if seed is provided
    if seed is not None:
        logger.info(f"Shuffling dataset with seed={seed}")
        dataset = dataset.shuffle(seed=seed)
    
    # Debug mode: truncate dataset
    if debug:
        dataset = dataset.select(range(min(debug_size, len(dataset))))
        logger.info(f"Debug mode: using only {len(dataset)} examples")
    
    # Prepare arguments for each shard
    list_args = []
    for shard_idx in range(num_workers):
        out = f'{output_path}_shard{shard_idx}_of_{num_workers}'.replace('/', '_')
        dst = f".cache/{out}"
        
        # Prepare function arguments
        func_args = {
            'dataset': dataset,
            'shard_idx': shard_idx,
            'total_shards': num_workers,
            'output_path': dst,
        }
        
        # Add the process function
        func_args['process_func'] = process_func
        
        # Add additional kwargs
        if process_func_kwargs:
            func_args.update(process_func_kwargs)
        
        list_args.append(func_args)
    
    # Process shards in parallel
    total_items = len(dataset)
    logger.info(f"Processing {total_items:,} examples using {num_workers} workers...")
    
    # Enable item-level progress tracking for Ray backend
    multi_process_kwargs = {
        'workers': num_workers,
        'backend': backend,
    }
    if backend == 'ray':
        multi_process_kwargs['total_items'] = total_items
        multi_process_kwargs['desc'] = f"Processing {total_items:,} items"
        multi_process_kwargs['poll_interval'] = 0.3
    
    output_paths = multi_process(
        _process_shard_wrapper, 
        list_args,
        **multi_process_kwargs
    )
    
    # Concatenate shards
    from datasets import concatenate_datasets, load_from_disk

    logger.info("Merging shards...")
    tmp_shards = [load_from_disk(p) for p in output_paths]
    merged_dataset = concatenate_datasets(tmp_shards)
    
    # Save final dataset
    final_size = len(merged_dataset)
    final_name = f"{Path(output_path).name}_size{final_size}"
    final_dst = Path(output_path).parent / final_name
    
    if final_dst.exists():
        logger.warning(f"Removing existing dataset: {final_dst}")
        shutil.rmtree(final_dst)
    
    logger.info(f"Saving final merged dataset to: {final_dst}")
    merged_dataset.save_to_disk(str(final_dst))
    
    # Cleanup temporary shards
    for p in output_paths:
        logger.info(f'Removing temporary shard: {p}')
        shutil.rmtree(p)
    
    logger.info(f"✅ Successfully processed dataset: {final_dst}")
    return str(final_dst)


def _process_shard_wrapper(args: Dict[str, Any]) -> str:
    """
    Wrapper function for processing a single shard.
    
    This wrapper extracts the dataset, shards it, and passes it to the user-provided
    process function along with any additional arguments.
    
    Args:
        args: Dictionary containing:
            - dataset: The full dataset
            - shard_idx: Index of the current shard
            - total_shards: Total number of shards
            - output_path: Path to save the processed shard
            - process_func: The function to apply to the shard
            - Additional arguments for the process function
            
    Returns:
        str: Path to the processed shard
        
    Note:
        Progress tracking is automatically available via report_progress()
        when using Ray backend with item-level tracking enabled.
    """
    from datasets import Dataset
    
    # Extract core parameters
    dataset = args.pop('dataset')
    shard_idx = args.pop('shard_idx')
    total_shards = args.pop('total_shards')
    output_path = args.pop('output_path')
    process_func = args.pop('process_func')
    
    # Remove progress_actor from args (it's in thread-local context now)
    args.pop('progress_actor', None)
    
    # Shard the dataset (HF datasets.shard() is memory-efficient)
    shard = dataset.shard(num_shards=total_shards, index=shard_idx)
    logger.info(f"Processing shard {shard_idx+1}/{total_shards} with {len(shard)} examples")
    
    # Process the shard with remaining kwargs
    # User code can call report_progress() directly for centralized tracking
    return process_func(shard, output_path, **args)
