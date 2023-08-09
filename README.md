# from speedy_utils

**from speedy_utils** is a fast and easy-to-use package for data science, designed to streamline various common tasks in Python programming and data analysis.

## Features

- Efficient utilities for caching and memoization.
- Handy functions for IO operations like JSON and pickle handling.
- Tools to assist with multi-threading and multi-processing tasks.
- Well-documented and easy to use.

## Installation

You can install `speedy-utils` via pip:

```bash
pip install speedy-utils
```

### Requirements

This package requires Python 3.6 or higher and the following packages:

- numpy
- requests
- xxhash
- loguru
- fastcore
- debugpy
- ipywidgets
- jupyterlab
- ipdb
- scikit-learn
- matplotlib
- pandas
- tabulate
- pydantic

These will be installed automatically when you install `speedy-utils`.

## Usage

Here’s a quick example of how to use the features of `speedy-utils`.

### Example: Using the Clock

```python
from speedy_utils import Clock

# Create an instance of Clock
clock = Clock()

# Start the clock
clock.start()

# ... some time-consuming operations ...

# Stop the clock
elapsed_time = clock.stop()
print(f'Time taken: {elapsed_time} seconds')
```

### Example: Using Memoization

```python
from speedy_utils import memoize

@memoize
def expensive_function(arg):
    # Simulate an expensive operation
    return arg * 2

result = expensive_function(10)
print(result)  # 20
```

## Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**AnhVTH**

- Email: anhvth.226@gmail.com
- GitHub: [anhvth](https://github.com/anhvth/speedy)
```

### Notes on Modifications

- Make sure to adjust any sections based on the specific features or functionalities of your package that you want to highlight.
- If you have a `LICENSE` file in your project, you can link to it properly in the License section.
- Feel free to add additional sections like "Testing" or "FAQ" if you think they would be useful for users.