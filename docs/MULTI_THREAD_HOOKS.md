# `multi_thread` completion hooks

`multi_thread` can process each completed item immediately without waiting for
the full worker pool to drain. Hooks run in the coordinator thread while worker
threads continue processing other items.

## Streaming results to a sink

```python
from speedy_utils import multi_thread


def save_result(index, item, result):
    writer.write(result)
    if writer.rows_since_checkpoint >= 10_000:
        writer.flush_and_checkpoint()


multi_thread(
    process_item,
    items,
    workers=200,
    ordered=False,
    prefetch_factor=2,
    on_result=save_result,
    collect=False,
)
```

`collect=False` prevents the result list from growing in memory and makes
`multi_thread` return `None`. It cannot be combined with
`store_output_pkl_file`.

## Hook contract

```python
def on_result(index, item, result): ...
def on_error(index, item, exception): ...
def on_progress(completed, total, succeeded, failed): ...
```

- `on_result` runs once for each successful logical item as soon as its future
  completes.
- `on_error` runs once for each failed logical item. Existing `error_handler`
  behavior still decides whether execution raises or continues.
- `on_progress` runs after each completed future. With `batch > 1`, one call
  can account for several logical items.
- All hooks run serially in the coordinator thread, so a hook can safely write
  to one buffered JSONL stream without a lock.
- Hook timing follows completion order even when `ordered=True`. `ordered`
  controls only the returned list.
- Exceptions raised by hooks stop the run. Keep hooks fast and move network or
  compute-heavy work into the worker function.

Worker threads continue running while the coordinator handles a hook. Small
buffered writes are normally negligible. Perform expensive durability calls
such as `fsync` only at checkpoint boundaries.

Hooks and `collect=False` also work with `n_proc > 1`. In that mode each child
process runs its own thread pool, then the parent coordinator delivers results
and progress as process chunks complete. Consequently hook updates are
chunk-granular rather than individual-future-granular.

```python
multi_thread(
    process_item,
    items,
    n_proc=10,
    workers=64,
    on_result=save_result,
    collect=False,
)
```

## Errors and checkpoints

```python
completed_jobs = set()


def save_batch(index, job, result):
    job_index, rows = result
    writer.write_rows(rows)
    completed_jobs.add(job_index)
    if writer.should_checkpoint(10_000):
        writer.commit(completed_jobs)


def record_failure(index, job, error):
    error_log.write(index=index, job=job, error=error)
```

The application owns output formatting and resume state. `multi_thread` owns
concurrency, completion delivery, progress reporting, and worker error policy.
