# Speedy Utils

![PyPI](https://img.shields.io/pypi/v/speedy-utils)
![Python Versions](https://img.shields.io/pypi/pyversions/speedy-utils)
![License](https://img.shields.io/pypi/l/speedy-utils)

**Speedy Utils** is a Python utility library for caching, parallel processing,
file I/O, LLM integration, dataset inspection, and image processing. The repo
ships multiple importable packages and keeps import time under the repository's
`0.4s` hook budget by keeping heavy external dependencies lazy.

## Table of Contents

- [Installation](#installation)
- [Packages](#packages)
- [Core Utilities](#core-utilities)
- [CLI Tools](#cli-tools)
- [LLM](#llm)
- [Dataset Tools](#dataset-tools)
- [Vision Utils](#vision-utils)
- [Testing and Checks](#testing-and-checks)

## Installation

```bash
pip install speedy-utils
# or
uv pip install speedy-utils
```

Install from source:

```bash
pip install git+https://github.com/anhvth/speedy_utils
# or
uv pip install git+https://github.com/anhvth/speedy_utils
```

Local development:

```bash
git clone https://github.com/anhvth/speedy_utils
cd speedy_utils
uv sync
```

Upgrading from older split packages:

```bash
pip uninstall speedy_llm_utils speedy_utils
pip install -U speedy-utils
```

## Packages

The wheel currently installs four packages from `src/`:

| Package | Purpose |
|---------|---------|
| `speedy_utils` | Core utilities: caching, I/O, formatting, parallelism, timing |
| `llm_utils` | OpenAI-compatible LLM wrappers and chat-format helpers |
| `vision_utils` | Image loading, plotting, and mmap-backed image datasets |
| `datasets_utils` | Dataset inspection helpers, including the `viz_chat` CLI |

## Core Utilities

### `memoize` and `imemoize`

```python
from speedy_utils import memoize, imemoize

@memoize
def expensive_function(x):
    import time

    time.sleep(2)
    return x * x


@imemoize
def fast_function(x):
    return x + 1
```

`memoize` uses memory, disk, or both. The default disk cache root is
`~/.cache/speedy_cache`.

```python
@memoize(
    keys=["x"],
    cache_dir="/tmp/my_cache",
    cache_type="both",   # "memory" | "disk" | "both"
    size=512,
    verbose=True,
)
def fn(x, ignored_arg):
    ...
```

Both decorators support sync and async functions.

### `multi_thread`

```python
from speedy_utils import multi_thread

results = multi_thread(lambda x: x * 2, [1, 2, 3, 4, 5])
```

Important public options:

```python
multi_thread(
    func,
    inputs,
    workers=None,
    batch=1,
    ordered=True,
    progress=True,
    progress_update=10,
    progress_total=None,
    progress_weight=None,
    prefetch_factor=4,
    timeout=None,
    error_handler="raise",   # "raise" | "ignore" | "log"
    max_error_files=100,
    store_output_pkl_file=None,
    **fixed_kwargs,
)
```

Error handling:

```python
def process(item):
    if item == 3:
        raise ValueError("bad item")
    return item * 2


multi_thread(process, [1, 2, 3], error_handler="raise")
multi_thread(process, [1, 2, 3], error_handler="ignore")
multi_thread(process, [1, 2, 3], error_handler="log")
```

`error_handler="log"` writes rich error reports under
`.cache/speedy_utils/error_logs/`.

### `multi_process`

```python
from speedy_utils import multi_process

results = multi_process(
    func,
    items,
    num_procs=4,
    num_threads=1,
    backend="spawn",      # "spawn" | "fork"
    error_handler="log",  # "raise" | "ignore" | "log"
    progress=True,
    dump_in_thread=True,
    log_worker="first",   # "zero" | "first" | "all"
)
```

Current behavior worth knowing:

- `num_procs=None` normalizes to `1`, not automatic process-count detection.
- `num_procs <= 1` and `num_threads <= 1` uses a local sequential backend.
- `num_procs <= 1` and `num_threads > 1` uses the in-process thread backend.

### File I/O

Use `load_jsonl()` for JSONL and `load_json_or_pickle()` for `.json` and pickle.

```python
from speedy_utils import (
    dump_json_or_pickle,
    dump_jsonl,
    jdumps,
    jloads,
    load_by_ext,
    load_json_or_pickle,
    load_jsonl,
)

records = load_jsonl("data/file.jsonl")
records = load_jsonl("data/**/*.jsonl")
records = load_jsonl(["train/*.jsonl", "val/file.jsonl"])

data = load_json_or_pickle("data.json")
data = load_json_or_pickle("data.pkl")

dump_json_or_pickle({"name": "Alice"}, "out.json")
dump_jsonl([{"a": 1}, {"a": 2}], "out.jsonl")

obj = jloads('{"key": "value",}')
text = jdumps(obj)

data = load_by_ext("data.csv")
data = load_by_ext(["part1.jsonl", "part2.jsonl"])
```

For streaming or compressed JSONL, use `fast_load_jsonl` directly:

```python
from speedy_utils.common.utils_io import fast_load_jsonl

for record in fast_load_jsonl(
    "data/large.jsonl.gz",
    progress=True,
    on_error="skip",
    max_lines=1000,
    use_orjson=True,
):
    ...
```

### Data, Printing, and Timing Helpers

```python
from speedy_utils import (
    Clock,
    convert_to_builtin_python,
    dedup,
    flatten_dict,
    flatten_list,
    fprint,
    print_table,
    timef,
)

flatten_list([[1, 2], [3, 4]])
flatten_dict({"a": {"b": 1}, "c": 2})
dedup([3, 1, 2, 1, 3])

fprint({"name": "Dana", "scores": [95, 87, 92]})
print_table([{"a": 1, "b": 2}, {"a": 3, "b": 4}])

@timef
def slow_function():
    ...

clock = Clock()
```

## CLI Tools

The installed console scripts are:

| CLI | Purpose |
|-----|---------|
| `mpython` | Launch sharded Python runs across tmux windows |
| `kill-mpython` | Kill `mpython` tmux sessions |
| `sp_chat` | Launch a Chainlit chat UI for an OpenAI-compatible backend |
| `spu-prefetch-large-model` | Read large model files into the OS page cache |
| `viz_chat` | Inspect chat datasets from JSON, JSONL, folders, or HF saves |
| `openapi_client_codegen` | Generate a sync client from an OpenAPI JSON spec |

Examples:

```bash
mpython -t 8 script.py
kill-mpython

sp_chat client=8000
sp_chat client=http://10.0.0.3:8000/v1 port=5010 model=Qwen/Qwen2.5-7B-Instruct

spu-prefetch-large-model /path/to/model -j 8

viz_chat data/my_dataset.jsonl
viz_chat data/hf_dataset/ --count 5
viz_chat data/tokenized_dataset/ --tokenizer Qwen/Qwen3-8B

openapi_client_codegen openapi.json -o generated_client.py
```

## LLM

`llm_utils` wraps OpenAI-compatible chat and completion APIs.

### `LLM` main entry points

```python
from llm_utils import LLM

llm = LLM(client=8000)
```

The three main sync entry points are:

- `chat_completion(...)` for chat responses.
- `generate(...)` for raw prompt continuation through the completions API.
- `pydantic_parse(...)` for structured outputs.

The convenience `llm(...)` wrapper routes like this:

- `llm("prompt")` -> `chat_completion(...)`
- `llm("prompt", response_model=MyModel)` -> `pydantic_parse(...)`
- `llm("prompt", return_dict=True)` -> normalized dict with raw artifacts

### Basic chat completion

```python
from llm_utils import LLM

llm = LLM(model="gpt-4o-mini")
message = llm("What is Python?")
print(message.content)
```

Equivalent explicit call:

```python
message = llm.chat_completion(
    [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is Python?"},
    ]
)
```

### Structured output with Pydantic

```python
from pydantic import BaseModel
from llm_utils import LLM


class Sentiment(BaseModel):
    sentiment: str
    confidence: float


llm = LLM(model="gpt-4o-mini")
result = llm.pydantic_parse(
    "Return JSON for the sentiment of: I love this product!",
    response_model=Sentiment,
)
print(result.sentiment, result.confidence)
```

### Normalized dict output

```python
result = llm(
    "Return JSON for the sentiment of: I love this product!",
    response_model=Sentiment,
    return_dict=True,
)

print(result.keys())
# dict_keys(["completion", "message", "messages", "parsed"])
```

### Streaming chat responses

Streaming is only supported for text completions, not Pydantic parsing.

```python
from llm_utils import LLM

llm = LLM(model="gpt-4o-mini")

for chunk in llm("Tell me a story", stream=True):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### Raw prompt continuation with `generate()`

`generate()` uses the completions API and returns an OpenAI
`CompletionChoice`-like object.

```python
choice = llm.generate(
    "Write a haiku about coding:",
    max_tokens=50,
    temperature=0.8,
)

print(choice.text)
print(choice.finish_reason)
print(choice.usage.total_tokens)
```

Current public behavior:

- `generate()` expects `prompt` to be a string.
- `n=1` only; multi-choice generation is rejected.
- backend-specific metadata such as `token_ids` or `prompt_logprobs` is kept
  when the backend returns it.

### Client configuration

```python
from llm_utils import LLM
from openai import OpenAI

llm = LLM(
    client=OpenAI(base_url="http://localhost:8000/v1", api_key="sk-..."),
    model="llama-3",
)

llm = LLM(client=8000, model="llama-3")
llm = LLM(client="http://localhost:8000/v1", model="llama-3")
llm = LLM(client=[8000, 8001, 8002], model="llama-3")
```

### Caching and history inspection

```python
llm = LLM(model="gpt-4o-mini", cache=True)

message = llm("What is 2+2?")
again = llm("What is 2+2?")
fresh = llm("What is 2+2?", cache=False)

history = llm.inspect_history()
```

`inspect_history()` returns the recent conversation that was recorded for the
last response.

### `LLMSignature`

`LLMSignature` binds a `Signature` class to default structured output.

```python
from llm_utils import Input, LLMSignature, Output, Signature


class SentimentSignature(Signature):
    text: str = Input("Text to analyze")
    sentiment: str = Output("positive | negative | neutral")
    confidence: float = Output("Confidence score")


sig = LLMSignature(signature=SentimentSignature, model="gpt-4o-mini")
result = sig("Analyze: I love this!")
print(result.sentiment, result.confidence)
```

### `Qwen3LLM`

`Qwen3LLM` adds staged prefix continuation for Qwen3-style reasoning flows.

Standard chat path:

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8000)
message = llm.chat_completion(
    [{"role": "user", "content": "Solve x^2 + 2x + 1 = 0"}],
    thinking_max_tokens=32,
    content_max_tokens=128,
)

print(message.content)
print(getattr(message, "reasoning_content", None))
print(getattr(message, "call_count", None))
```

Custom staged prefix flow:

```python
memory_state = llm.complete_until(
    [{"role": "user", "content": "Plan the answer in stages"}],
    "<memory>",
    stop="</memory>",
    max_tokens=128,
)

think_state = llm.complete_until(
    [{"role": "user", "content": "Plan the answer in stages"}],
    memory_state.assistant_prompt_prefix + "\n<think_efficient>",
    stop="</think_efficient>",
    max_tokens=256,
)

final_state = llm.complete_until(
    [{"role": "user", "content": "Plan the answer in stages"}],
    think_state.assistant_prompt_prefix,
    stop="<|im_end|>",
    max_tokens=256,
)

print(final_state.generated_text)
print(final_state.assistant_prompt_prefix)
print(final_state.call_count)
```

`complete_until()` returns a continuation state object, not a
`ChatCompletionMessage`.

## Dataset Tools

`datasets_utils.viz_chat` is a lightweight dataset inspector for conversation
data.

Supported inputs:

- HuggingFace datasets saved with `save_to_disk()`
- JSONL files
- JSON files containing one object or a list of objects
- Folders of JSON files
- tokenized datasets when `--tokenizer` is provided

Examples:

```bash
viz_chat data/my_dataset
viz_chat data/conversations.jsonl
viz_chat data/sharegpt.jsonl --format sharegpt
viz_chat data/tokenized_dataset/ --tokenizer Qwen/Qwen3-8B
viz_chat data/with_tools.jsonl --show-tools
```

## Vision Utils

`vision_utils` exports:

- `read_images`
- `read_images_cpu`
- `read_images_gpu`
- `plot_images_notebook`
- `ImageMmap`
- `ImageMmapDynamic`

### Image loading

The image loaders return a dict mapping each input path to a NumPy array or
`None` on failure.

```python
from vision_utils import read_images, read_images_cpu, read_images_gpu

paths = ["img1.jpg", "img2.png"]

images = read_images(paths)
cpu_images = read_images_cpu(paths)
gpu_images = read_images_gpu(paths)

first_image = images[paths[0]]
```

### Notebook plotting

`plot_images_notebook()` accepts NumPy arrays, PyTorch tensors, lists, or tuples
of image arrays. If you loaded images with `read_images*`, pass the values.

```python
from vision_utils import plot_images_notebook, read_images

paths = ["img1.jpg", "img2.png"]
images = read_images(paths)

plot_images_notebook(list(images.values()))
```

The current defaults include `dpi=300`, automatic grid sizing, and automatic
format normalization for `(H, W)`, `(H, W, C)`, `(C, H, W)`, `(B, H, W, C)`,
and `(B, C, H, W)` inputs.

### Mmap-backed datasets

Both mmap dataset classes take image paths, not a prebuilt mmap filename as the
only positional argument.

```python
from vision_utils import ImageMmap, ImageMmapDynamic

paths = ["img1.jpg", "img2.jpg"]

fixed = ImageMmap(paths, size=(224, 224))
dynamic = ImageMmapDynamic(paths)

img = fixed[0]
img2 = dynamic[0]
```

## Testing and Checks

```bash
# Run all tests with xdist workers
./tools/uv_test.sh -n 32

# Single test file
./tools/uv_test.sh tests/test_thread.py

# Verbose
./tools/uv_test.sh -v

# Check import-time budget
uv run python scripts/debug_import_time.py speedy_utils llm_utils vision_utils \
    --max-total-sec 0.4 --top 12 --min-sec 0.01 --no-stdlib

# Type checking
uv run python tools/check_syntax.py

# Ruff
uv run ruff check .
uv run ruff format .
```

### TDD and Regression Testing

- Every bug fix should include a regression test that reproduces the bug first.
- Keep regression tests deterministic (no flaky time/random/network behavior).
- Test through public APIs and assert specific outcomes.
- Keep one behavior per test so failures are easy to diagnose.
- Prefer fast, isolated tests that are run frequently.

See the full playbook in [`docs/TDD.md`](docs/TDD.md).
