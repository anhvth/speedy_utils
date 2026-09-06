# Item stages

Existing `process(item)` subclasses keep working. To opt into durable stages,
define `process(self, item, ctx)` and pass `context_root` and `semantic_config`
to `run_jsonl`. Keep the job class importable for spawn workers.

```python
class Job(ParallelLLMJob):
    def process(self, item, ctx):
        pair = ctx.stage("generate", lambda: self.generate(item))
        review = ctx.stage("review", lambda: self.review(pair))
        if not review["acceptable"]:
            ctx.reject("quality", review["reason"])
        return ctx.stage("audio", lambda: self.synthesize(pair, ctx),
                         retries=2, retry_if=lambda exc: isinstance(exc, TimeoutError))

job.run_jsonl(items, "run/rows.jsonl", target_rows=1000,
              context_root="run", semantic_config={"prompt_version": 1},
              checkpoint_every=50, status_interval=15)
```

Stage results must be JSON-serializable; Pydantic models become dictionaries.
Keys include the input item, semantic configuration, stage name, and previous
stage results. Change semantic configuration when prompts, model identity, or
generation settings change. Execution concurrency and equivalent endpoint routing
do not belong in semantic configuration. Increasing target_rows remains supported.

An interrupted item reuses its completed stages when replayed. A failed item is
recorded and replaced, following the existing job contract. Explicit rejection is
never retried. Retry predicates select transient exceptions only; defaults do not
retry. Avoid nesting these retries around clients that already retry internally.

`ctx.save_artifact("audios/sample.wav", data)` atomically writes bytes under the
context root with an item-specific filename and returns a relative path. Absolute
paths and parent traversal are rejected. The caller owns content validation.
Checkpoints are JSON files under `.stages/`; status snapshots are under `.status/`.
Retain checkpoints to resume. Run only one writer per output directory.

Live status reports committed rows, throughput, ETA, failures, rejections, and
per-stage active counts, average duration, retries and failures. Stage snapshots
are approximate and refreshed on stage events, not a GPU-utilization measure.
The ordered JSONL sink still waits for earlier items; stage checkpoints persist
completed work independently. Adaptive tuning and per-stage concurrency limits
are not implemented by this API.
