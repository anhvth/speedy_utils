import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .process import multi_process


if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

__all__ = ['multi_process_dataset']


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return max(1, (os.cpu_count() or 1) - 2)
    if num_workers <= 0:
        raise ValueError('num_workers must be a positive integer')
    return num_workers


def _prepare_dataset(
    dataset: 'Dataset',
    *,
    seed: int | None,
    debug: bool,
    debug_size: int,
) -> 'Dataset':
    if seed is not None:
        logger.info('Shuffling dataset with seed=%s', seed)
        dataset = dataset.shuffle(seed=seed)

    if debug:
        dataset = dataset.select(range(min(debug_size, len(dataset))))
        logger.info('Debug mode: using only %s examples', len(dataset))

    return dataset


def _build_shard_output_path(output_path: str, shard_idx: int, total_shards: int) -> str:
    shard_name = f'{output_path}_shard{shard_idx}_of_{total_shards}'.replace('/', '_')
    return str(Path('.cache') / shard_name)


def _build_shard_jobs(
    *,
    dataset: 'Dataset',
    output_path: str,
    process_func: Callable[..., str],
    process_func_kwargs: dict[str, Any],
    num_workers: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for shard_idx in range(num_workers):
        job = {
            'dataset': dataset,
            'shard_idx': shard_idx,
            'total_shards': num_workers,
            'output_path': _build_shard_output_path(output_path, shard_idx, num_workers),
            'process_func': process_func,
        }
        if process_func_kwargs:
            job.update(process_func_kwargs)
        jobs.append(job)
    return jobs


def _run_shard_jobs(jobs: list[dict[str, Any]], dataset_size: int, num_workers: int) -> list[str]:
    logger.info(
        'Processing %s examples using %s workers...',
        f'{dataset_size:,}',
        num_workers,
    )
    return multi_process(
        _process_shard_wrapper,
        jobs,
        num_procs=num_workers,
        backend='spawn',
        desc=f'Processing {dataset_size:,} items',
    )


def _final_output_path(output_path: str, dataset_size: int) -> Path:
    final_name = f'{Path(output_path).name}_size{dataset_size}'
    return Path(output_path).parent / final_name


def _merge_shards(output_paths: list[str], output_path: str) -> str:
    from datasets import concatenate_datasets, load_from_disk

    logger.info('Merging shards...')
    shard_datasets = [load_from_disk(path) for path in output_paths]
    merged_dataset = concatenate_datasets(shard_datasets)  # type: ignore[arg-type]

    final_path = _final_output_path(output_path, len(merged_dataset))
    if final_path.exists():
        logger.warning('Removing existing dataset: %s', final_path)
        shutil.rmtree(final_path)

    logger.info('Saving final merged dataset to: %s', final_path)
    merged_dataset.save_to_disk(str(final_path))
    return str(final_path)


def _cleanup_shards(output_paths: list[str]) -> None:
    for path in output_paths:
        logger.info('Removing temporary shard: %s', path)
        shutil.rmtree(path)


def multi_process_dataset(
    dataset: 'Dataset',  # type: ignore[name-defined]
    process_func: Callable,
    output_path: str,
    process_func_kwargs: dict[str, Any] | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    debug: bool = False,
    debug_size: int = 10000,
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
    num_workers = _resolve_num_workers(num_workers)
    dataset = _prepare_dataset(
        dataset,
        seed=seed,
        debug=debug,
        debug_size=debug_size,
    )

    shard_jobs = _build_shard_jobs(
        dataset=dataset,
        output_path=output_path,
        process_func=process_func,
        process_func_kwargs=process_func_kwargs or {},
        num_workers=num_workers,
    )
    output_paths = _run_shard_jobs(shard_jobs, len(dataset), num_workers)
    final_path = _merge_shards(output_paths, output_path)
    _cleanup_shards(output_paths)

    logger.info('Successfully processed dataset: %s', final_path)
    return final_path


def _process_shard_wrapper(args: dict[str, Any]) -> str:
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

    """
    job_args = dict(args)
    dataset = job_args.pop('dataset')
    shard_idx = job_args.pop('shard_idx')
    total_shards = job_args.pop('total_shards')
    output_path = job_args.pop('output_path')
    process_func = job_args.pop('process_func')
    job_args.pop('progress_actor', None)

    shard = dataset.shard(num_shards=total_shards, index=shard_idx)
    logger.info(
        'Processing shard %s/%s with %s examples',
        shard_idx + 1,
        total_shards,
        len(shard),
    )
    return process_func(shard, output_path, **job_args)
