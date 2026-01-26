# Speedy Utils

![PyPI](https://img.shields.io/pypi/v/speedy-utils)
![Python Versions](https://img.shields.io/pypi/pyversions/speedy-utils)
![License](https://img.shields.io/pypi/l/speedy-utils)

**Speedy Utils** is a Python utility library designed to streamline common programming tasks such as caching, parallel processing, file I/O, and data manipulation. It provides a collection of decorators, functions, and classes to enhance productivity and performance in your Python projects.

## 🚀 Recent Updates (January 27, 2026)

**Enhanced Error Handling in Parallel Processing:**

- Rich-formatted error tracebacks with code context and syntax highlighting
- Three error handling modes: 'raise', 'ignore', and 'log'
- Filtered tracebacks focusing on user code (hiding infrastructure)
- Real-time progress reporting with error/success statistics
- Automatic error logging to timestamped files
- Caller frame information showing where parallel functions were invoked

## Quick Start

### Parallel Processing with Error Handling

```python
from speedy_utils import multi_thread, multi_process

# Simple parallel processing
results = multi_thread(lambda x: x * 2, [1, 2, 3, 4, 5])
# Results: [2, 4, 6, 8, 10]

# Robust processing with error handling
def process_item(item):
    if item == 3:
        raise ValueError(f"Cannot process item {item}")
    return item * 2

# Continue processing despite errors
results = multi_thread(process_item, [1, 2, 3, 4, 5], error_handler='log')
# Results: [2, 4, None, 8, 10] - errors logged automatically
```

## Table of Contents

- [🚀 Recent Updates](#-recent-updates-january-27-2026)
- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Parallel Processing](#parallel-processing)
  - [Enhanced Error Handling](#enhanced-error-handling)
  - [Caching](#caching)
  - [File I/O](#file-io)
  - [Data Manipulation](#data-manipulation)
  - [Utility Functions](#utility-functions)
- [Testing](#testing)

## Features

- **Caching Mechanisms**: Disk-based and in-memory caching to optimize function calls.
- **Parallel Processing**: Multi-threading, multi-processing, and asynchronous multi-threading utilities with enhanced error handling.
- **File I/O**: Simplified JSON, JSONL, and pickle file handling with support for various file extensions.
- **Data Manipulation**: Utilities for flattening lists and dictionaries, converting data types, and more.
- **Timing Utilities**: Tools to measure and log execution time of functions and processes.
- **Pretty Printing**: Enhanced printing functions for structured data, including HTML tables for Jupyter notebooks.
- **Enhanced Error Handling**: Rich error tracebacks with code context, configurable error handling modes ('raise', 'ignore', 'log'), and detailed progress reporting.

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

### Extras

Optional dependencies can be installed via extras. For the `ray` backend
support (requires Python >= 3.9):

```bash
# pip
pip install 'speedy-utils[ray]'

# Poetry (for developing this repo)
poetry install -E ray
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

#### Multi-threading with Enhanced Error Handling

Execute functions concurrently using multiple threads with comprehensive error handling. The enhanced error handling provides three modes: 'raise' (default), 'ignore', and 'log'. When errors occur, you'll see rich-formatted tracebacks with code context and caller information.

```python
from speedy_utils import multi_thread

def process_item(item):
    # Simulate processing that might fail
    if item == 3:
        raise ValueError(f"Invalid item: {item}")
    return item * 2

items = [1, 2, 3, 4, 5]

# Default behavior: raise on first error with rich traceback
try:
    results = multi_thread(process_item, items, workers=3)
except SystemExit:
    print("Error occurred and was displayed with rich formatting")

# Continue processing on errors, return None for failed items
results = multi_thread(process_item, items, workers=3, error_handler='ignore')
print(results)  # [2, 4, None, 8, 10]

# Log errors to files and continue processing
results = multi_thread(process_item, items, workers=3, error_handler='log', max_error_files=10)
print(results)  # [2, 4, None, 8, 10] - errors logged to .cache/speedy_utils/error_logs/
```

#### Multi-processing with Error Handling

Process items across multiple processes with the same enhanced error handling capabilities.

```python
from speedy_utils import multi_process

def risky_computation(x):
    """Computation that might fail for certain inputs."""
    if x % 5 == 0:
        raise RuntimeError(f"Cannot process multiples of 5: {x}")
    return x ** 2

data = list(range(12))

# Process with error logging (continues on errors)
results = multi_process(
    risky_computation, 
    data, 
    backend='mp',
    error_handler='log',
    max_error_files=5
)
print(results)  # [0, 1, 4, 9, 16, None, 36, 49, 64, 81, None, 121]
```

### Enhanced Error Handling

**Speedy Utils** now provides comprehensive error handling for parallel processing with rich formatting and detailed diagnostics.

#### Rich Error Tracebacks

When errors occur, you'll see beautifully formatted tracebacks with:

- **Code context**: Lines of code around the error location
- **Caller information**: Shows where the parallel function was invoked
- **Filtered frames**: Focuses on user code, hiding infrastructure details
- **Color coding**: Easy-to-read formatting with syntax highlighting

#### Error Handling Modes

Choose how to handle errors in parallel processing:

- **`'raise'` (default)**: Stop on first error with detailed traceback
- **`'ignore'`**: Continue processing, return `None` for failed items  
- **`'log'`**: Log errors to files and continue processing

#### Error Logging

When using `error_handler='log'`, errors are automatically saved to timestamped files in `.cache/speedy_utils/error_logs/` with full context and stack traces.

#### Progress Reporting with Error Statistics

Progress bars now show real-time error and success counts:

```bash
Multi-thread [8/10] [00:02<00:00, 3.45it/s, success=8, errors=2, pending=0]
```

This makes it easy to monitor processing health at a glance.

#### Example: Robust Data Processing

```python
from speedy_utils import multi_thread

def process_data_record(record):
    """Process a data record that might have issues."""
    try:
        # Your processing logic here
        value = record['value'] / record['divisor']
        return {'result': value, 'status': 'success'}
    except KeyError as e:
        raise ValueError(f"Missing required field in record: {e}")
    except ZeroDivisionError:
        raise ValueError("Division by zero in record")

# Sample data with some problematic records
data = [
    {'value': 10, 'divisor': 2},     # OK
    {'value': 15, 'divisor': 0},     # Will error
    {'value': 20, 'divisor': 4},     # OK
    {'value': 25},                   # Missing divisor - will error
]

# Process with error logging - continues despite errors
results = multi_thread(
    process_data_record, 
    data, 
    workers=4,
    error_handler='log',
    max_error_files=10
)

print("Results:", results)
# Output: Results: [{'result': 5.0, 'status': 'success'}, None, {'result': 5.0, 'status': 'success'}, None]
# Errors are logged to files for later analysis
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

| Field     | Value                                 |
| --------- | ------------------------------------- |
| from_peft | ./outputs/llm_hn_qw32b/hn_results_r3/ |

Please ensure your code adheres to the project's coding standards and includes appropriate tests.
