# Speedy Utils

![PyPI](https://img.shields.io/pypi/v/speedy-utils)
![Python Versions](https://img.shields.io/pypi/pyversions/speedy-utils)
![License](https://img.shields.io/pypi/l/speedy-utils)

**Speedy Utils** is a Python utility library designed to streamline common programming tasks such as caching, parallel processing, file I/O, and data manipulation. It provides a collection of decorators, functions, and classes to enhance productivity and performance in your Python projects.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Caching](#caching)
  - [Parallel Processing](#parallel-processing)
  - [File I/O](#file-io)
  - [Data Manipulation](#data-manipulation)
  - [Utility Functions](#utility-functions)
- [Testing](#testing)

## Features

- **Caching Mechanisms**: Disk-based and in-memory caching to optimize function calls.
- **Parallel Processing**: Multi-threading, multi-processing, and asynchronous multi-threading utilities.
- **File I/O**: Simplified JSON, JSONL, and pickle file handling with support for various file extensions.
- **Data Manipulation**: Utilities for flattening lists and dictionaries, converting data types, and more.
- **Timing Utilities**: Tools to measure and log execution time of functions and processes.
- **Pretty Printing**: Enhanced printing functions for structured data, including HTML tables for Jupyter notebooks.

## Installation

You can install **Speedy Utils** via [PyPI](https://pypi.org/project/speedy-utils/) using `pip`:

```bash
uv pip install speedy-utils

```

Alternatively, install directly from the repository:

```bash
uv pip install git+https://github.com/anhvth/speedy
cd speedy-utils
pip install .
```

## Updating from previous versions

To update from previous versions or switch to v1.x, first uninstall any old
packages, then install the latest version:

```bash
pip uninstall speedy_llm_utils speedy_utils
pip install -e ./  # for local development
# or
pip install speedy_utils -U  # for PyPI upgrade
```

## Usage

Below are examples demonstrating how to utilize various features of **Speedy Utils**.

### Caching

#### Memoize Decorator

Cache the results of function calls to disk to avoid redundant computations.

```python
from speedy_utils import memoize

@memoize
def expensive_function(x):
    # Simulate an expensive computation
    import time
    time.sleep(2)
    return x * x

result = expensive_function(4)  # Takes ~2 seconds
result = expensive_function(4)  # Retrieved from cache instantly
```

#### In-Memory Memoization

Cache function results in memory for faster access within the same runtime.

```python
from speedy_utils import imemoize

@imemoize
def compute_sum(a, b):
    return a + b

result = compute_sum(5, 7)  # Computed and cached
result = compute_sum(5, 7)  # Retrieved from in-memory cache
```

### Parallel Processing

#### Multi-threading

Execute functions concurrently using multiple threads. This approach is straightforward and automatically handles both notebook and Python script executions. In a notebook environment, it delegates the running thread to a separate process. If interrupted, it immediately stops this process, avoiding thread dependency issues where threads continue running until all tasks are completed.

```python
from speedy_utils import multi_thread

def process_item(item):
    # Your processing logic
    return item * 2

items = [1, 2, 3, 4, 5]
results = multi_thread(process_item, items, workers=3)
print(results)  # [2, 4, 6, 8, 10]
```

### File I/O

#### Dumping Data

Save data in JSON, JSONL, or pickle formats.

```python
from speedy_utils import dump_json_or_pickle, dump_jsonl

data = {"name": "Alice", "age": 30}

# Save as JSON
dump_json_or_pickle(data, "data.json")

# Save as JSONL
dump_jsonl([data, {"name": "Bob", "age": 25}], "data.jsonl")

# Save as Pickle
dump_json_or_pickle(data, "data.pkl")
```

#### Loading Data

Load data based on file extensions.

```python
from speedy_utils import load_json_or_pickle, load_by_ext

# Load JSON
data = load_json_or_pickle("data.json")

# Load JSONL
data_list = load_json_or_pickle("data.jsonl")

# Load Pickle
data = load_json_or_pickle("data.pkl")

# Load based on extension with parallel processing
loaded_data = load_by_ext(["data.json", "data.pkl"])
```

### Data Manipulation

#### Flattening Lists and Dictionaries

```python
from speedy_utils import flatten_list, flatten_dict

nested_list = [[1, 2], [3, 4], [5]]
flat_list = flatten_list(nested_list)
print(flat_list)  # [1, 2, 3, 4, 5]

nested_dict = {"a": {"b": 1, "c": 2}, "d": 3}
flat_dict = flatten_dict(nested_dict)
print(flat_dict)  # {'a.b': 1, 'a.c': 2, 'd': 3}
```

#### Converting to Built-in Python Types

```python
from speedy_utils import convert_to_builtin_python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Charlie", age=28)
builtin_user = convert_to_builtin_python(user)
print(builtin_user)  # {'name': 'Charlie', 'age': 28}
```

### Utility Functions

#### Pretty Printing

```python
from speedy_utils import fprint, print_table

data = {"name": "Dana", "age": 22, "city": "New York"}

# Pretty print as table
fprint(data)

# Print as table using tabulate
print_table(data)
```

#### Timing Utilities

```python
from speedy_utils import timef, Clock

@timef
def slow_function():
    import time
    time.sleep(3)
    return "Done"

result = slow_function()  # Prints execution time

# Using Clock
clock = Clock()
# ... your code ...
clock.log()
```

## Testing

The project includes a comprehensive test suite using `unittest`. To run the tests, execute the following command in the project root directory:

```bash
python test.py
```

Ensure all dependencies are installed before running tests:

```bash
pip install -r requirements.txt
```

Run the script to parse and display the arguments:

```bash
python speedy_utils/common/dataclass_parser.py
```

Example output:

| Field              | Value                                 |
|--------------------|---------------------------------------|
| from_peft          | ./outputs/llm_hn_qw32b/hn_results_r3/ |

Please ensure your code adheres to the project's coding standards and includes appropriate tests.
