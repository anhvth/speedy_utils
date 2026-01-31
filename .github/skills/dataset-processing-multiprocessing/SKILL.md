---
name: 'dataset-processing-multiprocessing'
description: 'Advanced pattern for processing large HuggingFace datasets in parallel using speedy_utils.multi_process, with sharding, worker initialization, temporary file management, and safe cleanup.'
---

# Dataset Processing with Multiprocessing

This skill provides comprehensive guidance for processing large datasets (HuggingFace, Arrow, JSONL) in parallel using `speedy_utils.multi_process`. It covers the complete architectural pattern including worker design, data sharding, temporary file management, and robust error handling.

## When to Use This Skill

Use this skill when you need to:
- Process large HuggingFace datasets that don't fit efficiently in memory
- Apply expensive transformations (tokenization, format conversion, packing)
- Leverage multiple CPU cores for data preprocessing pipelines
- Combine dataset loading, transformation, and saving in one pipeline
- Integrate external tools (tokenizers, Megatron, format converters) with parallel processing
- Manage intermediate temporary files safely across multiple workers

**Ideal for:**
- Tokenization & packing pipelines (language model data prep)
- Format conversions (HuggingFace → Arrow → Format-specific)
- Image/audio processing with datasets
- Large-scale data cleaning and filtering
- Multi-stage pipelines with temporary outputs

## Why Multiprocessing for Dataset Processing

### Advantages

| Aspect | Benefit |
|--------|---------|
| **Throughput** | Process multiple shards simultaneously using all CPU cores |
| **Memory Isolation** | Each worker has independent memory; avoids single-process memory limits |
| **Scalability** | Scales with shard count and worker count independently |
| **I/O Parallelism** | Multiple workers reading/writing simultaneously improves disk throughput |
| **External Tools** | Easy integration with tokenizers, libraries that release the GIL |

### Comparison to Alternatives

| Approach | Use Case |
|----------|----------|
| **streaming.Dataset** | ✅ Small data, single-pass reads |
| **multi_thread** | ✅ I/O-bound operations (API calls, downloads) |
| **multi_process** | ✅ CPU-bound transformations, large datasets |
| **Ray/Dask** | ✅ Distributed computing across machines |

## Architecture: Complete Pattern

### Core Components

```
┌─────────────────────────────────────────┐
│ Main Process                            │
│  • Load dataset metadata                │
│  • Shard calculation                    │
│  • Argument preparation                 │
│  • multi_process dispatch               │
│  • Merge/finalize results               │
└────────────────────┬────────────────────┘
                     │
                     ├─ spawn ─┐
                     │         ├─ Worker 1 ──→ Shard 1
                     ├─ spawn ─┤ Worker 2 ──→ Shard 2
                     │         ├─ Worker 3 ──→ Shard 3
                     └─ spawn ─┤ Worker 4 ──→ Shard 4
                               └─ Worker N ──→ Shard N
                     
                     Each worker:
                     1. Load assigned shard range
                     2. Transform (tokenize, filter, convert)
                     3. Save to temporary location
                     4. Return result path
                     
                     Main process then:
                     5. Merge temporary results
                     6. Clean up temp files
                     7. Save final dataset
```

### Key Design Principles

1. **Stateless Workers**: Each worker function should be independent, with no shared state
2. **Auto-contained Imports**: Import dependencies inside worker function (not in main)
3. **Clear Data Flow**: Pass immutable arguments, return results only
4. **Temp File Hygiene**: Use unique temp directories per run, ensure cleanup
5. **Shard Independence**: No worker should depend on outputs of other workers
6. **Error Resilience**: Workers should handle failures gracefully, return None on error

## Step-by-Step Implementation Pattern

### Step 1: Define the Worker Function

```python
def process_shard(args):
    """
    Worker function processes one shard of data independently.
    
    Args:
        args: Tuple containing (shard_id, start_idx, end_idx, src_path, config...)
    
    Returns:
        Path to saved temporary result, or None if shard failed
    """
    shard_id, start_idx, end_idx, src_path, config = args
    
    # --- CRITICAL: Import inside worker ---
    import json
    from pathlib import Path
    from datasets import load_from_disk, Dataset
    
    # Setup paths for this worker
    temp_jsonl = Path(config['temp_dir']) / f"shard_{shard_id:05d}.jsonl"
    temp_arrow = Path(config['temp_dir']) / f"shard_{shard_id:05d}_arrow"
    
    try:
        # Load source dataset shard
        ds_local = load_from_disk(src_path)
        
        # STEP 1: Read & Transform
        valid_count = 0
        with open(temp_jsonl, 'w') as f:
            for i in range(start_idx, end_idx):
                try:
                    example = ds_local[i]
                    # Apply transformations
                    transformed = transform_example(example)
                    if transformed is not None:
                        f.write(json.dumps(transformed) + '\n')
                        valid_count += 1
                except Exception:
                    continue  # Skip problematic examples
        
        if valid_count == 0:
            # Clean up empty shard
            if temp_jsonl.exists():
                temp_jsonl.unlink()
            return None
        
        # STEP 2: Further processing (e.g., tokenization)
        tokenized_data = tokenize_jsonl(temp_jsonl, config['tokenizer_path'])
        
        # STEP 3: Save to arrow format
        ds = Dataset.from_dict(tokenized_data)
        ds.save_to_disk(str(temp_arrow))
        
        # Clean up intermediate JSONL
        if temp_jsonl.exists():
            temp_jsonl.unlink()
        
        return str(temp_arrow)  # Return path to result
        
    except Exception as e:
        # Log error and return None
        print(f"Shard {shard_id} failed: {e}")
        if temp_jsonl.exists():
            temp_jsonl.unlink()
        if temp_arrow.exists():
            import shutil
            shutil.rmtree(temp_arrow)
        return None
```

### Step 2: Prepare Worker Arguments

```python
import os
from pathlib import Path

def prepare_worker_arguments(src_path, total_records, num_workers, config):
    """
    Distribute data evenly across workers.
    
    Returns:
        List of argument tuples for each worker
    """
    worker_args = []
    rows_per_shard = total_records // num_workers
    
    for shard_id in range(num_workers):
        start_idx = shard_id * rows_per_shard
        
        # Last shard gets any remainder
        if shard_id == num_workers - 1:
            end_idx = total_records
        else:
            end_idx = start_idx + rows_per_shard
        
        args = (
            shard_id,
            start_idx,
            end_idx,
            str(Path(src_path).absolute()),  # IMPORTANT: absolute path for workers
            config  # Pass config dict with tokenizer_path, temp_dir, etc.
        )
        worker_args.append(args)
    
    return worker_args
```

### Step 3: Dispatch to multi_process

```python
from speedy_utils import multi_process
import shutil
from pathlib import Path
from datasets import concatenate_datasets, Dataset

def main(src_path, dst_path, num_workers=None):
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    
    # Setup temp directory
    temp_dir = (Path(dst_path).parent / f".tmp_{Path(dst_path).stem}").absolute()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load source to get metadata
    ds = load_from_disk(src_path)
    total_records = len(ds)
    
    # Prepare config
    config = {
        'temp_dir': str(temp_dir),
        'tokenizer_path': '/path/to/tokenizer',
        'seq_length': 2048,
    }
    
    try:
        # Prepare worker arguments
        worker_args = prepare_worker_arguments(
            src_path, total_records, num_workers, config
        )
        
        # Dispatch work
        shard_results = multi_process(
            process_shard,
            worker_args,
            workers=num_workers,
            backend='mp',  # Use multiprocessing for CPU work
            desc="Processing Shards",
        )
        
        # Filter out None results (failed workers)
        successful_shards = [r for r in shard_results if r is not None]
        
        # Merge results
        if successful_shards:
            merged_ds = concatenate_datasets([
                Dataset.load_from_disk(shard_path) 
                for shard_path in successful_shards
            ])
            merged_ds.save_to_disk(dst_path)
            print(f"✅ Saved {len(merged_ds)} records to {dst_path}")
        else:
            print("❌ All shards failed!")
    
    finally:
        # Always clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
```

## Complete Working Example

The following is a complete, production-ready example for tokenization and packing:

```python
import os
import sys
import json
import shutil
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from datasets import load_from_disk, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from speedy_utils import multi_process
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_shard(args):
    """
    Process one shard: format → tokenize → pack → save
    """
    shard_id, start_idx, end_idx, src_path, tokenizer_path, seq_length, temp_dir = args
    
    # Import inside worker
    import json
    import numpy as np
    from pathlib import Path
    from datasets import load_from_disk, Dataset
    from transformers import AutoTokenizer
    # Import megatron or other heavy libraries here
    
    shard_name = f"shard_{shard_id:05d}"
    temp_jsonl = os.path.join(temp_dir, f"{shard_name}.jsonl")
    temp_arrow = os.path.join(temp_dir, f"{shard_name}_arrow")
    
    try:
        # Load shard data
        ds_local = load_from_disk(src_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        valid_count = 0
        with open(temp_jsonl, 'w', encoding='utf-8') as f:
            for i in tqdm(range(start_idx, end_idx), desc=f"Format {shard_id}", leave=False):
                try:
                    ex = ds_local[i]
                    
                    # Filter & clean
                    if ex['messages'][-1]['role'] != 'assistant':
                        continue
                    
                    # Apply chat template
                    text = hf_tokenizer.apply_chat_template(
                        ex['messages'],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    # Split input/output
                    if "<|assistant|>" not in text:
                        continue
                    parts = text.split("<|assistant|>")
                    inp, out = parts[0], parts[-1]
                    
                    # Write
                    row = {'input': inp, 'output': out}
                    f.write(json.dumps(row) + '\n')
                    valid_count += 1
                except Exception:
                    continue
        
        if valid_count == 0:
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
            return None
        
        # Tokenize & Pack (using speedy_utils or custom logic)
        # ... tokenization code ...
        
        # Save packed data
        packed_records = [...]  # your packed data
        ds = Dataset.from_dict(packed_records)
        ds.save_to_disk(temp_arrow)
        
        # Cleanup
        if os.path.exists(temp_jsonl):
            os.remove(temp_jsonl)
        
        return temp_arrow
        
    except Exception as e:
        logger.error(f"Shard {shard_id} failed: {e}")
        # Cleanup on error
        for path in [temp_jsonl, temp_arrow]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source dataset path')
    parser.add_argument('--dst', required=True, help='Output dataset path')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer path')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=os.cpu_count() - 1)
    parser.add_argument('--backend', default='mp', choices=['mp', 'ray'])
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Setup temp dir
    temp_dir = Path(args.dst).parent / f".tmp_{Path(args.dst).stem}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        src_path = Path(args.src).absolute()
        dst_path = Path(args.dst).absolute()
        tokenizer_path = Path(args.tokenizer).absolute()
        
        ds = load_from_disk(str(src_path))
        total_rows = len(ds)
        num_shards = min(args.workers, total_rows)
        
        logger.info(f"Processing {total_rows} rows → {num_shards} shards")
        
        # Prepare worker args
        worker_args = []
        rows_per_shard = total_rows // num_shards
        for i in range(num_shards):
            start = i * rows_per_shard
            end = total_rows if i == num_shards - 1 else start + rows_per_shard
            worker_args.append((
                i, start, end,
                str(src_path),
                str(tokenizer_path),
                args.seq_len,
                str(temp_dir)
            ))
        
        # Dispatch
        results = multi_process(
            process_shard,
            worker_args,
            workers=args.workers,
            backend=args.backend,
            desc="Processing"
        )
        
        # Merge
        shard_paths = [r for r in results if r is not None]
        if shard_paths:
            full_ds = concatenate_datasets([
                Dataset.load_from_disk(p) for p in shard_paths
            ])
            full_ds.save_to_disk(str(dst_path))
            logger.info(f"✅ Saved {len(full_ds)} records to {dst_path}")
        else:
            logger.error("❌ No shards produced!")
        
        logger.info(f"Time: {time.time() - start_time:.2f}s")
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
```

## Best Practices

### 1. Absolute Paths Everywhere

Workers may run in different processes/machines. Always use absolute paths:

```python
# ✅ Good
src_path = Path(args.src).absolute()

# ❌ Bad
src_path = 'data/dataset'  # Relative path may not work in worker
```

### 2. Import Heavy Libraries Inside Workers

Avoid importing expensive libraries in the main process:

```python
# ✅ Good: Import inside worker
def process_shard(args):
    from transformers import AutoTokenizer
    ...

# ❌ Bad: Import at module level
from transformers import AutoTokenizer

def process_shard(args):
    ...
```

### 3. Shard Independence

Workers should not depend on each other's outputs:

```python
# ✅ Good: Each worker is independent
for i in range(num_shards):
    start = i * rows_per_shard
    end = (i + 1) * rows_per_shard
    process_shard(start, end)

# ❌ Bad: Workers depend on each other
for i in range(num_shards):
    process_and_depend_on_previous(i)
```

### 4. Graceful Degradation

Handle worker failures without crashing:

```python
# ✅ Good: Filter None results
results = multi_process(worker_func, args, ...)
successful = [r for r in results if r is not None]

# ❌ Bad: Assume all workers succeed
results = multi_process(worker_func, args, ...)
merged = concatenate_datasets(results)  # Crashes if any None
```

### 5. Cleanup in Finally Block

Always clean up temp files:

```python
temp_dir = Path('/tmp/processing')
try:
    # Do processing
    pass
finally:
    # Always runs, even on error
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
```

### 6. Worker Argument Design

Pass immutable, serializable data:

```python
# ✅ Good: Simple types that pickle well
args = (shard_id, start_idx, end_idx, src_path, config_dict)

# ❌ Bad: Non-serializable objects
args = (shard_id, dataset_object, complex_model)
```

### 7. Sharding Strategy

Balance load across workers:

```python
# ✅ Good: Distribute evenly
rows_per_shard = total_rows // num_workers
for i in range(num_workers):
    start = i * rows_per_shard
    end = total_rows if i == num_workers - 1 else (i+1) * rows_per_shard

# ❌ Bad: Uneven distribution
for i in range(num_workers):
    start = i * 1000  # Fixed size may leave last shard empty
    end = (i + 1) * 1000
```

## Performance Tuning

### Optimal Worker Count

```python
import os

# For CPU-bound work (tokenization, packing)
optimal_workers = max(1, os.cpu_count() - 1)

# For I/O-heavy work (reading/writing large files)
optimal_workers = os.cpu_count()  # Can use all cores
```

### Memory Considerations

```python
# Rule of thumb for memory per worker
memory_per_worker = total_memory // num_workers

# If worker exceeds memory, reduce num_workers:
while estimated_peak_memory > available_memory:
    num_workers -= 1
```

### Progress Monitoring

```python
from speedy_utils import multi_process

results = multi_process(
    worker_func,
    args,
    workers=num_workers,
    desc="Processing Shards",  # Shows TQDM progress bar
    backend='mp'
)
```

## Common Patterns

### Pattern 1: Incremental Processing

Process shards and save immediately (don't wait for merge):

```python
for shard_path in shard_results:
    if shard_path:
        # Process immediately
        ds = Dataset.load_from_disk(shard_path)
        # ... further processing ...
```

### Pattern 2: Data Filtering

Filter out invalid examples during processing:

```python
def process_shard(args):
    valid_count = 0
    with open(temp_file, 'w') as f:
        for i in range(start, end):
            example = ds[i]
            if is_valid(example):  # Filter here
                f.write(json.dumps(example) + '\n')
                valid_count += 1
    
    if valid_count == 0:
        return None  # Return None for empty shards
    return shard_path
```

### Pattern 3: Multi-Stage Processing

Apply multiple transformations in sequence:

```python
def process_shard(args):
    # Stage 1: Format
    formatted = format_data(raw_data)
    
    # Stage 2: Filter
    filtered = [x for x in formatted if is_valid(x)]
    
    # Stage 3: Tokenize
    tokenized = tokenize_batch(filtered)
    
    # Stage 4: Pack
    packed = pack_sequences(tokenized)
    
    # Save packed result
    save_to_disk(packed, temp_arrow)
    return temp_arrow
```

### Pattern 4: Error Recovery

Retry failed shards:

```python
failed_shards = [i for i, r in enumerate(results) if r is None]
if failed_shards:
    logger.warning(f"Retrying {len(failed_shards)} failed shards...")
    retry_args = [worker_args[i] for i in failed_shards]
    retry_results = multi_process(process_shard, retry_args, workers=1)
    # Merge retry results with original
```

## Error Handling

### Error Categories

| Error Type | Handling |
|------------|----------|
| **Data Corruption** | Skip example, log warning, continue |
| **OOM** | Reduce shard size, reduce seq_length |
| **Pickling Failure** | Ensure no non-serializable objects in args |
| **Worker Crash** | Return None, main process continues |
| **Missing Files** | Use absolute paths, verify before dispatch |

### Defensive Coding

```python
def process_shard(args):
    try:
        # ... processing ...
    except MemoryError:
        logger.error(f"OOM in shard {shard_id}")
        return None
    except Exception as e:
        logger.error(f"Shard {shard_id} failed: {str(e)[:100]}")
        return None
    finally:
        # Always cleanup
        cleanup_temp_files()
```

## Troubleshooting

### Problem: Workers hang or timeout

**Solution:** Add timeout parameter:
```python
results = multi_process(
    worker_func,
    args,
    timeout=600,  # 10 minutes per shard
    workers=num_workers
)
```

### Problem: Pickled file too large

**Solution:** Pass paths, not data:
```python
# ✅ Good: Pass path, load in worker
args = (shard_id, dataset_path, tokenizer_path)

# ❌ Bad: Pickle large dataset
args = (shard_id, large_dataset_object)
```

### Problem: Uneven shard sizes

**Solution:** Use last-shard handling:
```python
for i in range(num_shards):
    start = i * rows_per_shard
    if i == num_shards - 1:
        end = total_rows  # Last shard gets remainder
    else:
        end = start + rows_per_shard
```

### Problem: Temp directory cleanup fails (locked files on Windows)

**Solution:** Use absolute path and defer cleanup:
```python
import atexit
temp_dir = Path(...).absolute()
atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
```

## Integration with speedy_utils

### Using multi_process

```python
from speedy_utils import multi_process

results = multi_process(
    process_shard,
    worker_arguments,
    workers=num_workers,
    backend='mp',  # or 'ray', 'threadpool'
    desc="Processing",  # Progress bar description
    timeout=1200,  # Worker timeout in seconds
)
```

### Supported Backends

- **`mp`**: Multiprocessing (best for CPU-bound)
- **`ray`**: Ray cluster (distributed across machines)
- **`threadpool`**: Threading (only if I/O-bound within worker)
- **`safe`**: Fallback single-threaded (debugging)

## Bundled Assets

This skill includes practical resources:

### Example Script: `example_tokenize_pack.py`

A complete, runnable script demonstrating the full pattern with tokenization and packing.

**Usage:**
```bash
python example_tokenize_pack.py \
    --src /path/to/dataset \
    --dst /path/to/output \
    --tokenizer gpt2 \
    --workers 4
```

**Key features:**
- Proper error handling with cleanup
- Progress bars via tqdm
- Configurable tokenizer and sequence length
- Debug mode for testing
- Support for both `mp` and `ray` backends

### Configuration Reference: `CONFIG_REFERENCE.md`

Templates and guidelines for different processing scenarios:
- Standard tokenization configurations
- Large-scale distributed processing
- Memory-constrained environments
- Performance tuning tips
- Platform-specific notes

Quick start configurations for common use cases and performance optimization guidelines.

## References

- [speedy_utils.multi_process documentation](../../src/speedy_utils/__init__.py)
- [HuggingFace Datasets](https://huggingface.co/datasets/docs/about)
- [Arrow Format](https://arrow.apache.org/docs/)
- [Transformers Tokenization](https://huggingface.co/docs/transformers/tokenizer_summary)
