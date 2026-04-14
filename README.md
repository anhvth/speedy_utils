# Speedy Utils

![PyPI](https://img.shields.io/pypi/v/speedy-utils)
![Python Versions](https://img.shields.io/pypi/pyversions/speedy-utils)
![License](https://img.shields.io/pypi/l/speedy-utils)

**Speedy Utils** is a Python utility library for caching, parallel processing, file I/O, LLM integration, and image processing. It is designed for fast imports (< 0.4 s) via lazy loading of heavy dependencies.

## Table of Contents

- [Installation](#installation)
- [Packages](#packages)
- [Caching](#caching)
- [Parallel Processing](#parallel-processing)
  - [multi\_thread](#multi_thread)
  - [multi\_process](#multi_process)
  - [mpython CLI](#mpython-cli-tool)
- [File I/O](#file-io)
- [Data Utilities](#data-utilities)
- [Print & Display](#print--display)
- [Timing](#timing)
- [LLM](#llm)
  - [Basic completion](#basic-text-completion)
  - [Structured output](#structured-output-with-pydantic)
  - [Streaming](#streaming-responses)
  - [Client configuration](#client-configuration)
  - [Caching](#llm-caching)
  - [Signatures (DSPy-style)](#signatures-dspy-style)
  - [Qwen3LLM](#qwen3llm)
- [Vision Utils](#vision-utils)
- [Testing](#testing)

## Installation

```bash
pip install speedy-utils
# or
uv pip install speedy-utils
```

Install from source:

```bash
pip install git+https://github.com/anhvth/speedy
# or
uv pip install git+https://github.com/anhvth/speedy
```

Local development:

```bash
git clone https://github.com/anhvth/speedy
cd speedy
uv sync
```

Upgrading from an older split package:

```bash
pip uninstall speedy_llm_utils speedy_utils
pip install speedy-utils -U
```

## Packages

The repo ships three importable packages:

| Package | Purpose |
|---------|---------|
| `speedy_utils` | Core: caching, IO, parallel processing, timing |
| `llm_utils` | LLM integration with OpenAI-compatible backends |
| `vision_utils` | Image loading and visualization |

## Caching

### `@memoize` — disk + memory cache

```python
from speedy_utils import memoize

@memoize
def expensive_function(x):
    import time; time.sleep(2)
    return x * x

expensive_function(4)  # ~2 s, result saved to ~/.cache/speedy_utils/
expensive_function(4)  # instant, from cache
```

Options:

```python
@memoize(
    keys=["x"],          # which args contribute to the cache key
    cache_dir="/tmp/my_cache",
    cache_type="disk",   # "memory" | "disk" | "both" (default)
    size=512,            # LRU size for the in-memory layer
    verbose=True,
)
def fn(x, ignored_arg):
    ...
```

Works with `async` functions too.

### `@imemoize` — in-memory only

```python
from speedy_utils import imemoize

@imemoize
def compute_sum(a, b):
    return a + b

compute_sum(5, 7)  # computed
compute_sum(5, 7)  # from in-memory cache
```

The cache survives IPython `%load` re-executions (global persistent dict keyed on function source + args). Ideal for notebooks.

## Parallel Processing

### `multi_thread`

```python
from speedy_utils import multi_thread

results = multi_thread(lambda x: x * 2, [1, 2, 3, 4, 5])
# [2, 4, 6, 8, 10]
```

Full signature:

```python
multi_thread(
    func,
    inputs,
    *,
    workers=cpu_count() * 2,  # thread count
    batch=1,                  # items per invocation (1 = no batching)
    ordered=True,             # preserve input ordering
    progress=True,            # tqdm progress bar
    progress_weight=None,     # callable(item) → logical units
    prefetch_factor=4,        # in-flight work = workers * prefetch_factor
    timeout=None,             # overall wall-clock timeout (seconds)
    error_handler="raise",    # "raise" | "ignore" | "log"
    max_error_files=100,      # max error log files (log mode)
    store_output_pkl_file=None,  # persist results to this path
    **fixed_kwargs,           # forwarded to every func call
)
```

Error handling modes:

```python
def process(item):
    if item == 3:
        raise ValueError("bad item")
    return item * 2

# Stop on first error (default)
results = multi_thread(process, [1, 2, 3, 4, 5], error_handler="raise")

# Continue, return None for failed items
results = multi_thread(process, [1, 2, 3, 4, 5], error_handler="ignore")
# [2, 4, None, 8, 10]

# Log errors to .cache/speedy_utils/error_logs/ and continue
results = multi_thread(process, [1, 2, 3, 4, 5], error_handler="log")
# [2, 4, None, 8, 10]
```

Progress bars show live error/success counts:

```
Multi-thread [8/10] [00:02<00:00, 3.45it/s, success=8, errors=2]
```

### `multi_process`

```python
from speedy_utils import multi_process

results = multi_process(
    func,
    items,
    num_procs=4,         # process count (None = auto)
    num_threads=1,       # threads per process
    backend="spawn",     # "spawn" | "fork" (POSIX only)
    error_handler="log", # "raise" | "ignore" | "log"
    max_error_files=100,
    progress=True,
    desc=None,
    dump_in_thread=True, # persist results in a background thread
    log_worker="first",  # "zero" | "first" | "all"
)
```

Choosing `num_procs` vs `num_threads`:

| Workload | Recommended |
|----------|-------------|
| CPU-bound | `num_procs > 1`, `num_threads=1` (bypasses GIL) |
| I/O-bound | `num_procs=1`, `num_threads > 1` (lighter weight) |
| Mixed | Use both, e.g. `num_procs=4, num_threads=4` |

```python
# Web scraping: 4 processes for parsing, 8 threads each for I/O
results = multi_process(
    fetch_and_parse,
    urls,
    num_procs=4,
    num_threads=8,
    error_handler="log",
)
```

### mpython CLI Tool

Run a Python script across multiple tmux windows with automatic GPU/CPU allocation.

```bash
mpython script.py          # 16 workers across all GPUs
mpython -t 8 script.py    # 8 workers
mpython --gpus 0,1 script.py  # restrict to GPUs 0 and 1
kill-mpython               # kill all mpython sessions
```

Your script uses `MP_ID` / `MP_TOTAL` environment variables to shard work:

```python
import os

MP_ID    = int(os.getenv("MP_ID", "0"))
MP_TOTAL = int(os.getenv("MP_TOTAL", "1"))

inputs   = list(range(1000))
my_slice = inputs[MP_ID::MP_TOTAL]  # each worker gets its own slice

for item in my_slice:
    process(item)
```

Sessions are named `mpython`, `mpython-1`, `mpython-2`, … Attach with `tmux attach -t mpython`.

## File I/O

### Loading JSONL

```python
from speedy_utils import load_jsonl

# Single file
records = load_jsonl("data/file.jsonl")

# Glob pattern — all matches are concatenated into one list
records = load_jsonl("data/*.jsonl")

# Recursive glob
records = load_jsonl("data/**/*.jsonl")

# List of paths / globs
records = load_jsonl(["train/*.jsonl", "val/file.jsonl"])
```

For streaming (constant memory) or advanced options use `fast_load_jsonl` directly:

```python
from speedy_utils.common.utils_io import fast_load_jsonl

for record in fast_load_jsonl(
    "data/large.jsonl.gz",  # auto-detects .gz / .bz2 / .xz / .zst
    progress=True,
    on_error="skip",        # "raise" | "warn" | "skip"
    max_lines=1000,         # stop after N lines (sampling)
    use_orjson=True,        # faster parsing if orjson is installed
):
    ...
```

### JSON / Pickle

```python
from speedy_utils import dump_json_or_pickle, load_json_or_pickle

data = {"name": "Alice", "age": 30}

dump_json_or_pickle(data, "data.json")   # JSON
dump_json_or_pickle(data, "data.pkl")   # pickle

data = load_json_or_pickle("data.json")
data = load_json_or_pickle("data.pkl")
data = load_json_or_pickle("data.jsonl")  # returns list
```

### JSONL writing

```python
from speedy_utils import dump_jsonl

dump_jsonl([{"a": 1}, {"a": 2}], "out.jsonl")
```

### JSON helpers

```python
from speedy_utils import jdumps, jloads

s = jdumps({"key": "value"})   # json.dumps with indent=2, ensure_ascii=False

# jloads uses json_repair — tolerates malformed JSON
obj = jloads('{"key": "value",}')  # trailing comma fixed automatically
```

### Generic loader

```python
from speedy_utils import load_by_ext

# Dispatches by extension: .csv/.tsv → pandas, .txt → lines, .json/.pkl/…
data = load_by_ext("data.csv")
data = load_by_ext(["part1.jsonl", "part2.jsonl"])  # concatenated
data = load_by_ext("data.json", do_memoize=True)    # cached in memory
```

## Data Utilities

```python
from speedy_utils import flatten_list, flatten_dict, convert_to_builtin_python, dedup

# Flatten a list of lists
flatten_list([[1, 2], [3, 4], [5]])  # [1, 2, 3, 4, 5]

# Flatten nested dict with dot-notation keys
flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
# {"a.b": 1, "a.c": 2, "d": 3}

# Convert Pydantic models / numpy types to plain Python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

convert_to_builtin_python(User(name="Alice", age=30))
# {"name": "Alice", "age": 30}

# Deduplicate a list preserving order
dedup([3, 1, 2, 1, 3])  # [3, 1, 2]
```

## Print & Display

```python
from speedy_utils import fprint, print_table

data = {"name": "Dana", "scores": [95, 87, 92], "city": "New York"}

# Rich pretty-print (auto HTML in notebooks, grid in terminal)
fprint(data)
fprint(data, key_ignore=["scores"], grep="city")

# Tabular display (HTML in notebooks, grid in terminal)
print_table([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
```

## Timing

```python
from speedy_utils import timef, Clock

@timef
def slow_function():
    import time; time.sleep(3)

slow_function()  # prints execution time

# Checkpoint-based timer
clock = Clock()
do_step_one()
clock.log_elapsed_time()  # logs time since last checkpoint
do_step_two()
clock.log_elapsed_time()
```

## LLM

`llm_utils` provides a unified interface for OpenAI-compatible language model backends.

### Basic Text Completion

```python
from llm_utils import LLM

llm = LLM(model="gpt-4o-mini")

response = llm("What is Python?")
print(response.content)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from llm_utils import LLM

class Sentiment(BaseModel):
    sentiment: str
    confidence: float

llm = LLM(model="gpt-4o-mini")

result: Sentiment = llm.structured(
    "I love this product!",
    response_model=Sentiment,
)
print(result.sentiment, result.confidence)
```

### Streaming Responses

```python
from llm_utils import LLM

llm = LLM(model="gpt-4o-mini")

for chunk in llm("Tell me a story", stream=True):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### Client Configuration

```python
from llm_utils import LLM
from openai import OpenAI

# Custom OpenAI-compatible client
llm = LLM(client=OpenAI(base_url="http://localhost:8000/v1", api_key="sk-..."), model="llama-3")

# Port shorthand — wraps OpenAI(base_url="http://localhost:<port>/v1")
llm = LLM(client=8000, model="llama-3")

# URL shorthand
llm = LLM(client="http://localhost:8000/v1", model="llama-3")

# Load balancing across multiple clients
llm = LLM(client=[8000, 8001, 8002], model="llama-3")
```

### LLM Caching

```python
from llm_utils import LLM

llm = LLM(model="gpt-4o-mini", cache=True)  # default: True

result = llm("What is 2+2?")          # hits API
result = llm("What is 2+2?")          # served from cache
result = llm("What is 2+2?", cache=False)  # bypass cache for this call
```

### Signatures (DSPy-style)

`LLMSignature` provides a structured, declarative way to define LLM tasks:

```python
from llm_utils import LLMSignature, Signature, Input, Output, InputField, OutputField

class SentimentSignature(Signature):
    """Analyze sentiment of the given text."""
    text: str = InputField(description="Text to analyze")
    sentiment: str = OutputField(description="positive | negative | neutral")
    confidence: float = OutputField(description="Confidence score 0–1")

sig = LLMSignature(SentimentSignature, model="gpt-4o-mini")
result = sig(text="I love this!")
print(result.sentiment, result.confidence)
```

### Qwen3LLM

Extends `LLM` for Qwen3 models with staged generation and explicit thinking support:

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(model="Qwen/Qwen3-0.6B", enable_thinking=True)

# Stage 1: generate memory
mem = llm.complete_until(
    [{"role": "user", "content": "Solve this in stages"}],
    "<memory>",
    stop="</memory>",
    max_tokens=256,
)

# Stage 2: generate reasoning
think = llm.complete_until(
    [{"role": "user", "content": "Solve this in stages"}],
    mem.assistant_prompt_prefix + "\n<think>",
    stop="</think>",
    max_tokens=512,
)

# Stage 3: final answer
final = llm.complete_until(
    [{"role": "user", "content": "Solve this in stages"}],
    think.assistant_prompt_prefix,
    stop="<|im_end|>",
    max_tokens=512,
)

print(final.content)
```

## Vision Utils

```python
from vision_utils import read_images, read_images_cpu, read_images_gpu, plot_images_notebook

paths = ["img1.jpg", "img2.png"]

# Auto-select CPU or GPU loader
images = read_images(paths)

# Explicit CPU loader (Pillow)
images = read_images_cpu(paths)

# GPU loader (NVIDIA DALI — requires dali installed)
images = read_images_gpu(paths)

# Display in a Jupyter notebook
plot_images_notebook(images)
```

Memory-mapped image datasets for large collections:

```python
from vision_utils import ImageMmap, ImageMmapDynamic

dataset = ImageMmap("dataset.mmap")
img = dataset[0]  # zero-copy read
```

## Testing

```bash
# Run all tests with 32 workers
./tools/uv_test.sh -n 32

# Single test file
./tools/uv_test.sh tests/test_thread.py

# Verbose
./tools/uv_test.sh -v

# Check import time (must be < 0.4 s)
uv run python -c "import time; s=time.perf_counter(); import speedy_utils; print(f'{time.perf_counter()-s:.3f}s')"

# Detailed import budget analysis
uv run python scripts/debug_import_time.py speedy_utils llm_utils vision_utils \
    --max-total-sec 0.4 --top 12 --min-sec 0.01 --no-stdlib

# Type checking (zero pyright errors required before commit)
uv run python tools/check_syntax.py
```
