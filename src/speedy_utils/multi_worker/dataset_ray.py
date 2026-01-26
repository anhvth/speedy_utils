"""
Efficient Ray-based parallel processing for HuggingFace datasets.

This module provides a simple, dataset.map-like API that leverages Ray for
distributed processing while handling per-worker resource initialization
(like tokenizers) efficiently.
"""
import os
import time
import threading
from typing import Callable, Any, TypeVar, Generic
from pathlib import Path

__all__ = ['multi_process_dataset_ray', 'WorkerResources']


class WorkerResources:
    """
    Container for per-worker resources that should be initialized once per worker.
    
    Example:
        def init_worker():
            return WorkerResources(
                tokenizer=AutoTokenizer.from_pretrained("..."),
                model=load_model(),
            )
        
        def process_item(item, resources):
            tokens = resources.tokenizer.encode(item['text'])
            return {'tokens': tokens}
        
        results = multi_process_dataset_ray(process_item, dataset, worker_init=init_worker)
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def multi_process_dataset_ray(
    func: Callable[[Any, Any], Any],
    dataset,
    *,
    workers: int | None = None,
    worker_init: Callable[[], WorkerResources] | None = None,
    batch_size: int = 1,
    desc: str = "Processing",
    progress: bool = True,
    return_results: bool = True,
    output_path: str | Path | None = None,
    **func_kwargs,
) -> list[Any] | None:
    """
    Process a HuggingFace dataset in parallel using Ray.
    
    Simple API similar to dataset.map() but with Ray parallelism and efficient
    per-worker resource initialization.
    
    Args:
        func: Function to apply to each item. Signature: func(item, resources=None, **kwargs)
              where resources is the WorkerResources from worker_init (if provided).
        dataset: HuggingFace dataset (or path to dataset on disk).
        workers: Number of workers. None = use all available Ray CPUs.
        worker_init: Optional function that returns WorkerResources, called ONCE per worker.
                     Use this for expensive initialization like loading tokenizers/models.
        batch_size: Process items in batches for efficiency (default: 1 = per-item).
        desc: Description for progress bar.
        progress: Show progress bar.
        return_results: If True, collect and return all results. If False, return None
                        (useful when func writes to disk).
        output_path: If provided, save results to this path as they complete.
        **func_kwargs: Additional kwargs passed to func.
    
    Returns:
        List of results from func(item) for each item, or None if return_results=False.
    
    Example:
        # Simple usage
        results = multi_process_dataset_ray(
            lambda item, **_: item['text'].upper(),
            dataset
        )
        
        # With per-worker tokenizer
        def init_worker():
            from transformers import AutoTokenizer
            return WorkerResources(tokenizer=AutoTokenizer.from_pretrained("gpt2"))
        
        def tokenize(item, resources, max_length=512):
            return resources.tokenizer.encode(item['text'], max_length=max_length)
        
        results = multi_process_dataset_ray(
            tokenize,
            dataset,
            worker_init=init_worker,
            max_length=1024,
        )
    """
    import ray
    from tqdm import tqdm
    import numpy as np
    
    # Handle dataset path vs object
    dataset_path = None
    if isinstance(dataset, (str, Path)):
        dataset_path = str(dataset)
        import datasets
        dataset = datasets.load_from_disk(dataset_path)
    elif hasattr(dataset, '_data_files') or hasattr(dataset, '_indices'):
        # It's a HF dataset object - try to get its path
        # Workers will reload from disk for memory efficiency
        try:
            if hasattr(dataset, 'cache_files') and dataset.cache_files:
                dataset_path = str(Path(dataset.cache_files[0]['filename']).parent)
        except Exception:
            pass
    
    total_items = len(dataset)
    
    # Initialize Ray and get available workers
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    
    if workers is None:
        workers = int(ray.cluster_resources().get("CPU", os.cpu_count() or 4))
    
    # Ensure we don't have more workers than items
    workers = min(workers, total_items)
    
    # Pre-compute shard boundaries (avoid per-worker shuffle!)
    shard_size = total_items // workers
    shard_ranges = []
    for i in range(workers):
        start = i * shard_size
        end = total_items if i == workers - 1 else (i + 1) * shard_size
        shard_ranges.append((start, end))
    
    # Progress tracking actor
    @ray.remote
    class ProgressActor:
        def __init__(self, total):
            self.count = 0
            self.total = total
        
        def update(self, n=1):
            self.count += n
            return self.count
        
        def get_count(self):
            return self.count
    
    progress_actor = ProgressActor.remote(total_items) if progress else None
    
    # Define the worker task
    @ray.remote
    def process_shard(
        shard_id: int,
        start_idx: int,
        end_idx: int,
        dataset_path_or_ref,
        worker_init_fn,
        func_to_apply,
        func_kw,
        batch_sz,
        progress_ref,
        do_return,
    ):
        import datasets
        
        # Load dataset (memory-mapped = fast)
        if isinstance(dataset_path_or_ref, str):
            ds = datasets.load_from_disk(dataset_path_or_ref)
        else:
            ds = ray.get(dataset_path_or_ref)
        
        # Select this worker's slice
        shard = ds.select(range(start_idx, end_idx))
        del ds  # Free reference
        
        # Initialize per-worker resources ONCE
        resources = worker_init_fn() if worker_init_fn else None
        
        results = [] if do_return else None
        count = 0
        
        if batch_sz == 1:
            # Per-item processing
            for item in shard:
                result = func_to_apply(item, resources=resources, **func_kw)
                if do_return:
                    results.append(result)
                count += 1
                if progress_ref and count % 100 == 0:
                    ray.get(progress_ref.update.remote(100))
        else:
            # Batch processing
            batch = []
            for item in shard:
                batch.append(item)
                if len(batch) >= batch_sz:
                    batch_results = func_to_apply(batch, resources=resources, **func_kw)
                    if do_return:
                        if isinstance(batch_results, list):
                            results.extend(batch_results)
                        else:
                            results.append(batch_results)
                    count += len(batch)
                    if progress_ref and count % 100 < batch_sz:
                        ray.get(progress_ref.update.remote(min(100, len(batch))))
                    batch = []
            
            # Process remaining items
            if batch:
                batch_results = func_to_apply(batch, resources=resources, **func_kw)
                if do_return:
                    if isinstance(batch_results, list):
                        results.extend(batch_results)
                    else:
                        results.append(batch_results)
                count += len(batch)
        
        # Report remaining progress
        if progress_ref:
            remaining = count % 100
            if remaining > 0:
                ray.get(progress_ref.update.remote(remaining))
        
        return results
    
    # Put dataset in object store if no path available
    dataset_ref = dataset_path if dataset_path else ray.put(dataset)
    
    # Submit all shard tasks
    futures = []
    for i, (start, end) in enumerate(shard_ranges):
        future = process_shard.remote(
            i, start, end,
            dataset_ref,
            worker_init,
            func,
            func_kwargs,
            batch_size,
            progress_actor,
            return_results,
        )
        futures.append(future)
    
    # Progress bar polling thread
    pbar = None
    stop_polling = threading.Event()
    
    def poll_progress():
        nonlocal pbar
        pbar = tqdm(total=total_items, desc=desc, disable=not progress)
        while not stop_polling.is_set():
            try:
                count = ray.get(progress_actor.get_count.remote())
                pbar.n = count
                pbar.refresh()
            except Exception:
                pass
            stop_polling.wait(0.3)
        # Final update
        try:
            pbar.n = total_items
            pbar.refresh()
            pbar.close()
        except Exception:
            pass
    
    if progress and progress_actor:
        poll_thread = threading.Thread(target=poll_progress, daemon=True)
        poll_thread.start()
    
    # Collect results
    t0 = time.time()
    try:
        shard_results = ray.get(futures)
    finally:
        stop_polling.set()
        if progress and progress_actor:
            poll_thread.join(timeout=2)
    
    elapsed = time.time() - t0
    rate = total_items / elapsed if elapsed > 0 else 0
    print(f"✅ Processed {total_items:,} items in {elapsed:.1f}s ({rate:.1f} items/s)")
    
    if not return_results:
        return None
    
    # Flatten results from all shards
    all_results = []
    for shard_result in shard_results:
        if shard_result:
            all_results.extend(shard_result)
    
    # Optionally save
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"💾 Saved results to {output_path}")
    
    return all_results
