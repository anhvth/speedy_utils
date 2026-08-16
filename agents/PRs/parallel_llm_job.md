# Implement `ParallelLLMJob`

## Goal

Add a reusable class that lets other projects implement only item-specific LLM
logic while Speedy Utils owns LLM setup, parallel execution, hooks, progress,
errors, and optional JSONL checkpointing.

```python
from speedy_utils import ParallelLLMJob


class GenerateConversation(ParallelLLMJob[dict, Conversation]):
    def process(self, item):
        prompt = self.build_prompt(item)
        return self.llm(prompt, response_model=Conversation)


rows = GenerateConversation(client=7788, workers=200).run(items)
```

## Contract

The execution lifecycle is:

```text
raw item -> prepare(item) -> process(prepared) using self.llm
         -> finalize(raw item, result) -> hook/output
```

- `process(item)` is required.
- `prepare(item)` defaults to identity.
- `finalize(item, result)` defaults to returning `result`.
- `self.llm` is an `llm_utils.LLM` instance configured by the constructor.
- `run()` accepts arbitrary Python inputs and preserves output types.
- Thread mode is the default for LLM network I/O.
- Process mode must create its LLM lazily inside each process; never pickle a
  live client.
- `run_jsonl()` accepts Pydantic models or JSON-compatible outputs, converts
  Pydantic values with `model_dump(mode="json")`, and supports stable item IDs,
  resume, buffered writes, and durable checkpoints.

## Public API

```python
ParallelLLMJob(
    client,
    model=None,
    workers=32,
    mode="thread",
    prefetch_factor=2,
    error_handler="ignore",
    **llm_defaults,
)

job.run(
    items,
    ordered=True,
    collect=True,
    on_result=None,
    on_error=None,
    on_progress=None,
)

job.run_jsonl(
    items,
    output,
    checkpoint_every=10_000,
    resume=True,
    target_items=None,
    id_key="id",
)
```

Hooks run in the coordinator, not worker threads. Keep them fast. JSONL writes
are buffered; `flush` and `fsync` occur only at checkpoint boundaries.

## Files

- Add `src/speedy_utils/parallel_llm_job.py`.
- Export `ParallelLLMJob` lazily from `src/speedy_utils/__init__.py` without
  regressing import time.
- Reuse `llm_utils.LLM` and the existing `multi_thread` hook API; do not create
  another executor implementation.
- Add `tests/test_parallel_llm_job.py` with a fake LLM—no network calls.
- Add `docs/PARALLEL_LLM_JOB.md` with basic, structured-output, hook, JSONL,
  interruption, and resume examples.
- Update `/home/anhvth8/.agents/skills/speedy-utils-core/SKILL.md` to link to
  the new document.

## Acceptance criteria

- Users only need to subclass and implement `process(self, item)`.
- `self.llm` is available inside `prepare`, `process`, and `finalize`.
- `run(collect=False)` does not retain results in memory.
- Completion hooks execute before the entire pool drains.
- Ordered collection and completion-order hooks behave independently.
- JSONL resume never duplicates committed items and loses at most the active
  checkpoint interval after interruption.
- Thread errors follow the existing `error_handler` policy.
- Process mode initializes one lazy LLM per process.
- Existing `multi_thread`, `multi_process`, and LLM APIs remain compatible.
- Focused tests, syntax checks, Ruff, and import-time checks pass.
