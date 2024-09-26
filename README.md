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
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

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
pip install speedy-utils
```

Alternatively, install directly from the repository:

```bash
git clone https://github.com/yourusername/speedy-utils.git
cd speedy-utils
pip install .
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

Execute functions concurrently using multiple threads.

```python
from speedy_utils import multi_thread

def process_item(item):
    # Your processing logic
    return item * 2

items = [1, 2, 3, 4, 5]
results = multi_thread(process_item, items, workers=3)
print(results)  # [2, 4, 6, 8, 10]
```

#### Multi-processing

Leverage multiple CPU cores for parallel execution.

```python
from speedy_utils import multi_process

def compute_square(n):
    return n * n

numbers = list(range(10))
squares = multi_process(compute_square, numbers, workers=4)
print(squares)  # [0, 1, 4, 9, ..., 81]
```

#### Asynchronous Multi-threading

Combine asynchronous programming with multi-threading for efficient I/O-bound operations.

```python
import asyncio
from speedy_utils import async_multi_thread

def fetch_data(url):
    import requests
    response = requests.get(url)
    return response.text

urls = [
    "https://example.com",
    "https://openai.com",
    "https://github.com",
]

async def main():
    results = await async_multi_thread(fetch_data, urls, desc="Fetching URLs")
    for content in results:
        print(len(content))

asyncio.run(main())
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

## Deployment

The project is configured to publish releases to [PyPI](https://pypi.org/) using GitHub Actions. To publish a new version:

1. **Create a Git Tag**: Follow semantic versioning (e.g., `v0.1.0`).
2. **Push to Repository**: Push the tag to trigger the GitHub Actions workflow.

The workflow defined in `.github/workflows/publish.yml` will handle building and uploading the package to PyPI. Ensure you have set the `PYPI_API_TOKEN` in your repository secrets.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**: Click the "Fork" button at the top right of the repository page.
2. **Create a Branch**: 
    ```bash
    git checkout -b feature/YourFeature
    ```
3. **Commit Changes**: 
    ```bash
    git commit -m "Add your feature"
    ```
4. **Push to Fork**: 
    ```bash
    git push origin feature/YourFeature
    ```
5. **Create a Pull Request**: Navigate to the repository and create a pull request from your fork.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Happy Coding! 🚀**