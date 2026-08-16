# Parallel LLM jobs

`ParallelLLMJob` keeps application code focused on one item. The library owns
the LLM lifecycle, bounded parallelism, ordered output, failures, checkpoints,
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
Resume skips to the committed input index. For exact-target generation, make
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
- `prefetch_factor` bounds queued work per concurrency slot; results are never
  accumulated for the whole dataset.

Each child constructs one LLM pool on first use. A live client is never
pickled. Bare SSH endpoints such as `h1-31:8000` are resolved once in the
parent, so all children reuse the same parent-owned loopback forwards. HTTP(S)
URLs remain direct endpoints, and integers remain local ports.

Workers may finish out of order, but the coordinator writes in input order.
This gives deterministic row order and lets checkpoint state use one compact
input cursor instead of retaining every completed ID.

## Checkpoints and resume

`run_jsonl` creates three files:

- the requested JSONL output;
- `<output>.state.json`, containing counters, byte offsets, and the next input
  index;
- `<output>.errors.jsonl`, or the explicit `error_log` path.

At each checkpoint the coordinator flushes and fsyncs both JSONL files before
atomically replacing state. Resume truncates any uncommitted tails to the
recorded byte offsets and continues from the recorded input index. A non-empty
output without state is rejected rather than guessed, unless it already has
exactly the requested target row count.

A completed exact-size job may resume with a larger `target_rows`; it continues
from the saved input cursor and appends only the additional rows. Smaller or
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
