---
title: Centralized Progress Tracking for Distributed Processing
description: Automatic item-level progress tracking across distributed workers with clean user API
tags: [distributed, progress, ray, multiprocessing, speedy_utils]
difficulty: intermediate
use_cases:
  - Large dataset processing with 64+ workers
  - Token-level or item-level progress visibility in distributed tasks
  - Replacing worker-local tqdm bars with centralized tracking
---

# Centralized Progress Tracking for Distributed Processing

## Problem

When processing large datasets with many workers (e.g., 64 workers tokenizing 2.9M examples), traditional approaches create **one tqdm progress bar per worker**, resulting in **64 cluttered progress bars** flooding the terminal. This makes it impossible to see actual progress.

**Before (Cluttered - 64 progress bars):**
```
Format & Filter: 100%|███████| 46069/46069 [Worker 1]
Format & Filter: 100%|███████| 46070/46070 [Worker 2]
Tokenize: 45%|████████░░░| 21000/46069 [Worker 1]
Tokenize: 67%|████████████░| 31000/46070 [Worker 2]
... (60 more bars)
```

**After (Clean - Single centralized bar):**
```
Processing 2,948,390 items:  57%|████████████░░░░░| 1,671,800/2,948,390 [01:05<00:49, 25,607it/s]
```

## Solution

speedy_utils provides **automatic centralized progress tracking** via `report_progress()` - a simple function workers call to report items processed. All complexity is handled internally using Ray actors and thread-local storage.

## Implementation Pattern

### 1. User Code (Clean & Simple)

```python
from speedy_utils import report_progress

def process_items(dataset, output_path):
    """Your processing function - stays clean and minimal."""
    results = []
    
    for i, item in enumerate(dataset):
        # Do expensive work
        result = expensive_operation(item)
        results.append(result)
        
        # Report progress every 100 items
        if i % 100 == 0 and i > 0:
            report_progress(100)
    
    # Save results
    save_to_disk(results, output_path)
    return output_path
```

**That's it!** No:
- ❌ `progress_actor` parameters
- ❌ `progress_callback` wrappers  
- ❌ `if progress_actor is not None` checks
- ❌ Ray imports or `ray.get()` calls
- ❌ Callback threading complexity

### 2. Multi-Process Dataset Pattern

```python
from speedy_utils import multi_process_dataset

final_path = multi_process_dataset(
    dataset=my_dataset,
    process_func=process_items,
    output_path="./output",
    num_workers=64,
    backend='ray'  # Enables item-level tracking
)
```

The framework automatically:
- ✅ Creates centralized `ProgressActor`
- ✅ Injects progress context into worker threads
- ✅ Routes `report_progress()` to centralized tracker
- ✅ Shows single unified progress bar
- ✅ Updates in near-realtime (every 300ms)

## Real-World Example: SFT Dataset Packing

### Processing 2.9M Examples with 64 Workers

```python
# experiments/datasets/standalone_packer.py
from speedy_utils import report_progress

class SFTDatasetPacker:
    def _format_and_filter(self, dataset):
        from speedy_utils import report_progress
        
        for i in range(len(dataset)):
            # Apply chat template, filter, etc.
            formatted_item = self.tokenizer.apply_chat_template(...)
            
            # Report progress every 100 items
            if i % 100 == 0 and i > 0:
                report_progress(100)
    
    def _tokenize_all(self, formatted_data):
        from speedy_utils import report_progress
        
        for idx, example in enumerate(formatted_data):
            # Tokenize
            tokenized = self._tokenize_example(example)
            
            # Report progress
            if idx % 100 == 0 and idx > 0:
                report_progress(100)
    
    def _fill_bins(self, assignments, sequences):
        from speedy_utils import report_progress
        
        for idx, assignment in enumerate(assignments):
            # Pack sequences into bins
            packed = self._pack(assignment, sequences)
            
            # Report progress every 10 bins
            if idx % 10 == 0 and idx > 0:
                report_progress(10)
```

### Terminal Output

```bash
$ python standalone_packer.py --src ./data/SFT_merged_2.9M \
                               --dst ./data/packed \
                               --tokenizer tokenizers/Qwen3-32B \
                               --seq_len 16368 \
                               --workers 64

2026-02-01 11:34:02 - Loading dataset from: ./data/SFT_merged_2.9M
2026-02-01 11:34:03 - Processing 2,948,390 examples using 64 workers...

Processing 2,948,390 items:  57%|████████████░░░░░| 1,671,800/2,948,390 [01:05<00:49, 25,607it/s]
```

**Single clean progress bar showing:**
- Total items: 2,948,390
- Current progress: 1,671,800 (57%)
- Processing rate: 25,607 items/second
- Time elapsed/remaining

## How It Works (Internal)

### Architecture

```
User calls multi_process_dataset(backend='ray', total_items=2.9M)
    ↓
speedy_utils creates ProgressActor (Ray actor = centralized counter)
    ↓
Ray worker tasks receive progress_actor in shared_refs
    ↓
_task() wrapper auto-injects progress_actor into thread-local storage
    ↓
User code calls report_progress(100)
    ↓
report_progress() reads actor from thread-local storage
    ↓
Calls progress_actor.update.remote(100) (thread-safe Ray call)
    ↓
Main process polls ProgressActor every 300ms via ProgressPoller
    ↓
Updates tqdm bar: pbar.n = stats["processed"]; pbar.refresh()
```

### Thread-Local Storage Pattern

```python
# speedy_utils/multi_worker/progress.py
_progress_context = threading.local()

def set_progress_context(progress_actor):
    """Auto-called by speedy_utils when setting up workers."""
    _progress_context.actor = progress_actor

def report_progress(n: int = 1):
    """Public API - users call this directly."""
    actor = getattr(_progress_context, 'actor', None)
    if actor is not None:
        import ray
        ray.get(actor.update.remote(n))
```

### Auto-Injection in Ray Workers

```python
# speedy_utils/multi_worker/_multi_process_ray.py
@ray.remote
def _task(x, shared_refs_dict, regular_kwargs_dict):
    from .progress import set_progress_context
    
    # Extract progress_actor from shared refs
    progress_actor_ref = shared_refs_dict.get('progress_actor')
    
    # Auto-inject into thread-local storage
    if progress_actor_ref is not None:
        set_progress_context(progress_actor_ref)
    
    # User function runs with automatic progress tracking
    return user_function(x, **kwargs)
```

## Configuration

### Batched Reporting (Reduce Overhead)

```python
# Report every 100 items (recommended)
for i, item in enumerate(items):
    process(item)
    if i % 100 == 0 and i > 0:
        report_progress(100)

# For faster operations, report more frequently
for i, item in enumerate(fast_items):
    process(item)
    if i % 10 == 0:
        report_progress(10)

# For very slow operations, report per-item
for item in slow_items:
    expensive_operation(item)
    report_progress(1)
```

### Poll Interval (Update Frequency)

```python
multi_process_dataset(
    dataset=dataset,
    process_func=process_items,
    output_path="./output",
    num_workers=64,
    backend='ray',
    poll_interval=0.3  # Update every 300ms (default)
    # poll_interval=1.0  # Less frequent updates (lower overhead)
    # poll_interval=0.1  # More frequent updates (higher overhead)
)
```

## Performance Characteristics

| Aspect | Value | Notes |
|--------|-------|-------|
| **Overhead per call** | ~0.1ms | Ray remote call overhead |
| **Recommended batch size** | 100-1000 items | Balance between updates and overhead |
| **Update frequency** | 300ms | Main process polls every 300ms |
| **Acceptable lag** | 2-3 seconds | Due to batched updates |
| **Thread safety** | Built-in | Ray actor model ensures safety |
| **Memory overhead** | Minimal | Single actor for all workers |

## Backend Support

| Backend | Item-Level Tracking | Notes |
|---------|---------------------|-------|
| `ray` | ✅ Yes | Full support with ProgressActor |
| `mp` | ❌ No | ThreadPool - task-level only |
| `seq` | ❌ No | Sequential - task-level only |

**Calling `report_progress()` with non-Ray backends is safe** (no-op).

## When to Use

### ✅ Use Centralized Progress Tracking When:

- Processing datasets with **10+ workers**
- Operations are **item-level granular** (tokenizing, formatting, etc.)
- You want **visibility into actual progress**, not just task completion
- Processing **100K+ items** where progress matters
- Using **Ray backend** for distributed processing

### ❌ Don't Use When:

- Single worker or local execution (use local tqdm instead)
- Very fast operations (<1ms per item, overhead not worth it)
- Backend is not Ray
- You don't need progress tracking

## Troubleshooting

### Progress Bar Shows Weird Numbers

**Problem:** Bar shows `3376200it` instead of percentage
```
Processing 2,948,390 items[ray]: 3376200it [02:36, 21520.00it/s]
```

**Cause:** tqdm `total` not set correctly before item tracking starts

**Fix:** Ensure `total_items` is passed to `multi_process()` and tqdm is created with correct total upfront (fixed in implementation)

### Progress Not Updating

**Cause 1:** Not calling `report_progress()`
- **Fix:** Add `report_progress(n)` calls in your worker function

**Cause 2:** Calling too infrequently
- **Fix:** Reduce batch size (report every 10-100 items instead of 1000+)

**Cause 3:** Backend not Ray
- **Fix:** Set `backend='ray'` in `multi_process_dataset()`

### Multiple Progress Bars Still Appear

**Cause:** Worker functions contain local `tqdm()` calls
**Fix:** Remove/disable local tqdm bars when `report_progress()` is used

```python
# Bad: Local tqdm + centralized tracking
for i in tqdm(range(len(items))):  # Creates 64 bars!
    process(items[i])
    report_progress(1)

# Good: Use report_progress() only
for i in range(len(items)):
    process(items[i])
    if i % 100 == 0:
        report_progress(100)
```

## Backward Compatibility

✅ **100% backward compatible:**

- Calling `report_progress()` when not using Ray: **no-op** (safe)
- Calling `report_progress()` in single-worker mode: **no-op** (safe)
- Using `backend='mp'` or `backend='seq'`: **no change** (task-level progress)
- Not calling `report_progress()`: **still works** (task-level progress only)
- Old code without progress tracking: **unchanged**

## Testing

Simple test to verify centralized tracking:

```python
from speedy_utils import multi_process_dataset, report_progress
from datasets import Dataset, Features, Value

def process_shard(shard, output_path):
    """Test worker function with progress reporting."""
    import time
    
    results = []
    for i, item in enumerate(shard):
        time.sleep(0.001)  # Simulate work
        results.append({'text': item['text'].upper()})
        
        # Report progress every 100 items
        if i % 100 == 0 and i > 0:
            report_progress(100)
    
    # Save and return
    save_to_disk(results, output_path)
    return output_path

# Create test dataset
def gen():
    for i in range(10000):
        yield {'text': f'Sample {i}', 'id': i}

dataset = Dataset.from_generator(gen, features=Features({
    'text': Value('string'),
    'id': Value('int64')
}))

# Process with centralized tracking
final_path = multi_process_dataset(
    dataset=dataset,
    process_func=process_shard,
    output_path="/tmp/test_output",
    num_workers=8,
    backend='ray'
)

# Expected: Single progress bar showing "Processing 10,000 items: X%"
```

## Best Practices

1. **Report in batches** (100-1000 items) to reduce overhead
2. **Import at function level** to avoid circular imports:
   ```python
   def process_items(items):
       from speedy_utils import report_progress  # Import here
       ```
3. **Remove local tqdm bars** when using centralized tracking
4. **Use consistent batch sizes** across worker functions
5. **Don't report too frequently** (<10 items) - wastes overhead
6. **Don't report too infrequently** (>1000 items) - progress lags

## Related Skills

- [dataset-processing-multiprocessing](/skills/dataset-processing-multiprocessing/SKILL.md) - Pattern for processing HF datasets in parallel
- [multi-threading-processing](/skills/multi-threading-processing/SKILL.md) - When to use threads vs processes
- [megatron-bridge-config-system](/skills/megatron-bridge-config-system/SKILL.md) - Large-scale training configuration

## References

- **speedy_utils source:** `/home/anhvth8/projects/speedy_utils/src/speedy_utils/multi_worker/`
  - `progress.py` - ProgressActor and report_progress()
  - `_multi_process_ray.py` - Auto-injection logic
  - `dataset_sharding.py` - multi_process_dataset()
- **Example usage:** `/home/anhvth8/projects/SFT/experiments/datasets/standalone_packer.py`
- **Test script:** `/home/anhvth8/projects/SFT/test_centralized_progress.py`