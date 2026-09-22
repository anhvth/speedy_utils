# Parallel LLM jobs

`ParallelLLMJob` keeps application code focused on one item. The library owns
the LLM lifecycle, bounded parallelism, completion-order output, failures, checkpoints,
resume, and exact-size JSONL generation.

```python
from speedy_utils import ParallelLLMJob


class GenerateText(ParallelLLMJob[dict, dict]):
    def process(self, item):
        message = self.llm(item["prompt"], enable_thinking=False)
        text = str(message.content or "").strip()
        return {"id": item["id"], "text": text} if text else None


job = GenerateText(
    client=["h1-31:8000", "h2-14:8100"],
    processes=4,
    threads_per_process=32,
    cache=False,
)
summary = job.run_jsonl(
    items,
    "generated.jsonl",
    target_rows=500_000,
    checkpoint_every=10_000,
)
print(summary.written_rows, summary.elapsed_seconds)
```

## Subclass contract

Implement `process(item)`. Return the final JSON-serializable output row, or
`None` when an otherwise valid request should be rejected and replaced. An
exception is recorded in the error JSONL and is also replaced. The job layer
does not retry an item; `LLM` continues to retry transient HTTP failures.

Input items must expose a stable `id`, either as a mapping key or attribute.
Override `item_id(item)` only when the identifier lives elsewhere. Override
`iter_outputs(item, result)` when one input intentionally produces multiple
rows; the default yields one row.

The input iterable must be deterministic and replayable from its beginning.
Resume skips checkpointed inputs. For exact-target generation, make
the iterable large enough or unbounded so rejected inputs can be replaced.

Do not create global OpenAI or LLM clients in user code. `self.llm` is lazy,
thread-safe, process-local, and shared by the threads in its process. Keep
other subclass state picklable and treat it as read-only during `process()`.
Sequential jobs may intentionally share an existing pool with `llm=shared_llm`
when `processes=1`; injected live clients are rejected in process mode.

## Parallel execution

- `processes=1` uses an in-process thread pool.
- `processes>1` starts that many persistent spawn workers, each with
  `threads_per_process` long-lived threads.
- Effective request concurrency is `processes * threads_per_process`.
- `prefetch_factor` bounds unfinished submitted work per concurrency slot.
  Each completion frees a submission slot, even if an earlier input is slow.
  By default, received results are appended immediately and released. Only
  `ordered=True` buffers completed results behind earlier unfinished inputs;
  that buffer may accumulate beyond the prefetch limit.

Each child constructs one LLM pool on first use. A live client is never
pickled. Bare SSH endpoints such as `h1-31:8000` are resolved once in the
parent, so all children reuse the same parent-owned loopback forwards. HTTP(S)
URLs remain direct endpoints, and integers remain local ports.

New jobs default to `ordered=False`: one coordinator appends received results
directly to the final JSONL and flushes each item's rows for immediate visibility.
Each object row automatically receives `_input_index`, the zero-based position
in the original input iterable. Non-object results are wrapped as
`{"_input_index": 2, "output": ...}`. Pydantic models become objects first.
Fan-out rows share their input index. The field is reserved: a result containing
`_input_index` raises an error instead of overwriting it. Input objects are not mutated.

For example, input 2 can finish before input 0:

```jsonl
{"_input_index":2,"id":"third","text":"finished first"}
{"_input_index":0,"id":"first","text":"finished second"}
```

Use `ordered=True` to preserve input order and the original output schema.
Omitting `ordered` on resume preserves the checkpoint's mode, including legacy
ordered checkpoints. Explicitly changing a checkpoint's mode is rejected; use a
new output path for a new mode. The API uses `ordered=None` for this automatic
selection; it resolves to False for new jobs. With completion-order output,
the first received successes fill `target_rows`, so both order and selection
can vary between runs.

The tqdm bar shows saved rows, elapsed time, rate, and an ETA when a total is
known. With `progress_total`, it instead counts received input outcomes,
including errors and rejections, and shows `saved` separately. `pending` counts
submitted inputs without a received outcome (queued or running); `errors`
counts observed item errors and `rejected` counts processed rejections.
Only ordered jobs show `buffered`, when results are waiting for earlier inputs.
The display refreshes about once a second. Saved rows are visible in the file;
durability is established at checkpoints.

## Checkpoints and resume

`run_jsonl` writes only the requested JSONL in its output directory. Its
checkpoint and diagnostic files live in a stable workspace under
`/tmp/parallel_llm/<job-id>/`, where `job-id` is derived from the resolved
output path:

- the requested JSONL output;
- `state.json`, containing counters, byte offsets, and the next input index;
- `errors.jsonl`, or the explicit `error_log` path.
- `completed.jsonl` for unordered jobs, an append-only journal of finalized
  input indices, including failures and rejections that have no output row.

At each checkpoint the coordinator flushes and fsyncs the output, error log,
and completion journal before atomically replacing state with their byte offsets.
Unordered jobs checkpoint after `checkpoint_every` finalized inputs or
`flush_interval` seconds (default 10, checked by the coordinator), whichever
comes first. This avoids an fsync for every row. Resume truncates uncommitted
tails and loads a set of completed indices to skip during input replay. Ordered
jobs use a compact input cursor instead and checkpoint by count.
A non-empty output without state is rejected. Only explicit `ordered=True`
allows the legacy shortcut for an output already at exactly the requested target.

A completed exact-size job may resume with a larger `target_rows`; it continues
with uncommitted inputs and appends only the additional rows. Smaller or
otherwise incompatible target changes are rejected.

The sink writes exactly `target_rows`, even when `iter_outputs` fans out. If
the iterable ends first, `TargetNotReachedError` exposes the partial
`JobSummary`. Infrastructure, serialization, checkpoint, and worker-process
failures stop the run.

## Structured output

Pydantic results are serialized with `model_dump(mode="json")`:

```python
from pydantic import BaseModel
from speedy_utils import ParallelLLMJob


class Answer(BaseModel):
    id: str
    answer: str


class StructuredJob(ParallelLLMJob[dict, Answer]):
    def process(self, item):
        parsed = self.llm(item["prompt"], response_model=Answer)
        return parsed.model_copy(update={"id": item["id"]})
```

`JobSummary` reports attempted, succeeded, rejected, and failed inputs,
written rows, paths, resume status, and elapsed time. Re-running a completed
job with the same target returns immediately from its state file.

### Materialized results on cancellation

Each completed item is atomically saved under the output's temporary workspace
at `/tmp/parallel_llm/<job-id>/materialized/` before being returned by
its worker. Resume reuses these results, including completions waiting behind
an earlier unfinished item. Ordered JSONL and checkpoint cursors retain their
existing contract. Keep this workspace for resume; only replay identical inputs
and job code. Errors/rejections are also retained.
A fresh run with `resume=False` uses a new materialization namespace.
Existing checkpoints upgrade automatically; results lost from an older process's
memory cannot be recovered. Thread cancellation still waits for running calls;
already materialized work survives a subsequent forced termination.

### Indexed snapshots for finite collection

Use `run_jsonl(items, output, indexed=True, checkpoint_every=100,
flush_interval=10)` for one nonempty JSON object per input. The output starts
with one `{}` slot per input. Completions update slots in memory; atomic snapshots
flush by completion count or elapsed time without waiting for earlier rows.
Cancellation flushes received results. Forced termination can lose the unflushed
batch. Resume fills empty slots; increasing the input count appends empty slots.
Keep input order fixed. Existing compact output is imported by matching row IDs.

This mode holds inputs and results in memory and rewrites the JSONL snapshot
because rows have variable lengths. New runs avoid per-result files. Errors and
rejections leave empty slots for rerun. Fan-out outputs, exact-size replacement,
and stage-context jobs use the streaming mode described above (unordered by
default, or explicitly ordered). Indexed mode retains its separate slot-based
schema and resume contract.
