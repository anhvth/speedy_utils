# Vision Utils

Utility functions for computer vision tasks, particularly for visualizing images in Jupyter notebooks.

## Features

### `plot_images_notebook(images, ...)`

Plot a batch of images in a notebook with automatic format detection, smart grid layout, and size optimization to prevent notebook breaking.

**Supported Input Formats:**
- **Numpy arrays:**
  - Batch: `(B, H, W, C)` or `(B, C, H, W)`
  - Single: `(H, W, C)`, `(C, H, W)`, or `(H, W)` for grayscale
- **PyTorch tensors:**
  - Batch: `(B, H, W, C)` or `(B, C, H, W)`
  - Single: `(H, W, C)`, `(C, H, W)`, or `(H, W)` for grayscale
- **List of images:**
  - Mixed formats (numpy arrays and/or torch tensors)

**Parameters:**
- `images`: Images to plot (see supported formats above)
- `nrows`: Number of rows in grid (default: None, auto-calculated using sqrt)
- `ncols`: Number of columns in grid (default: None, auto-calculated using sqrt)
- `figsize`: Figure size `(width, height)` in inches (default: auto-calculated based on grid size and image count)
- `titles`: Optional list of titles for each image
- `cmap`: Colormap for grayscale images (default: 'gray')
- `dpi`: Dots per inch for the figure (default: 72)
- `max_figure_width`: Maximum figure width in inches (default: 15.0)
- `max_figure_height`: Maximum figure height in inches (default: 20.0)

**Smart Features:**
- **Auto Grid Layout**: When both `nrows` and `ncols` are None, uses sqrt to create roughly square grid
- **Adaptive Cell Sizing**: Automatically reduces cell size for larger image counts to prevent notebook breaking
- **Size Constraints**: Respects max width/height limits to ensure plots render properly in notebooks
- **Aspect Ratio Aware**: Calculates figure size based on actual image aspect ratios

## Installation

The vision_utils module is part of speedy_utils. Install with:

```bash
pip install -e .
```

For PyTorch support:
```bash
pip install torch
```

For plotting (required):
```bash
pip install matplotlib
```

## Usage Examples

### Auto Grid Layout (Recommended)

```python
import numpy as np
from vision_utils import plot_images_notebook

# Auto grid - creates 3x3 grid for 9 images
images = np.random.rand(9, 64, 64, 3)
plot_images_notebook(images)

# Auto grid - creates 3x3 grid for 8 images (1 empty cell)
images = np.random.rand(8, 64, 64, 3)
plot_images_notebook(images)
```

### Manual Grid Layout

```python
import numpy as np
from vision_utils import plot_images_notebook

# Specify exact grid dimensions
images = np.random.rand(8, 64, 64, 3)
plot_images_notebook(images, nrows=2, ncols=4)
```

### Basic Usage

```python
import numpy as np
from vision_utils import plot_images_notebook

# Batch of images - auto grid layout
images = np.random.rand(8, 64, 64, 3)
plot_images_notebook(images)
```

### Many Images (Adaptive Sizing)
```

### Many Images (Adaptive Sizing)

```python
import numpy as np
from vision_utils import plot_images_notebook

# Automatically uses smaller cells for many images
images = np.random.rand(25, 64, 64, 3)
plot_images_notebook(images)  # Creates 5x5 grid with optimized cell size
```

### PyTorch Tensors

```python
import torch
from vision_utils import plot_images_notebook

# PyTorch tensor in (B, C, H, W) format
images = torch.rand(8, 3, 64, 64)
plot_images_notebook(images)
```

### Mixed Formats

```python
import numpy as np
from vision_utils import plot_images_notebook

# List of images with different formats
images = [
    np.random.rand(64, 64, 3),  # (H, W, C)
    np.random.rand(3, 64, 64),  # (C, H, W)
    np.random.rand(64, 64),     # Grayscale
]

plot_images_notebook(images, titles=titles)
```

### Single Image

```python
import numpy as np
from vision_utils import plot_images_notebook

# Single image
image = np.random.rand(128, 128, 3)
plot_images_notebook(image)
```

### Custom Figure Size and Colormap

```python
import numpy as np
from vision_utils import plot_images_notebook

# Grayscale images with custom settings
images = np.random.rand(4, 64, 64)
plot_images_notebook(
    images,
    nrows=2,
    ncols=2,
    figsize=(8, 8),
    cmap='viridis',
    dpi=100
)
```

## Smart Grid Layout

When `nrows` and `ncols` are both `None` (default), the function automatically calculates the optimal grid layout:

1. Uses `sqrt(n_images)` to determine a roughly square grid
2. For 9 images → 3x3 grid
3. For 8 images → 3x3 grid (with 1 empty cell)
4. For 16 images → 4x4 grid
5. For 25 images → 5x5 grid

This ensures balanced layouts without manual calculation.

## Adaptive Cell Sizing

The function automatically adjusts cell sizes based on the number of images to prevent notebook breaking:

- **1-4 images**: 4.0 inches per cell
- **5-9 images**: 3.0 inches per cell
- **10-16 images**: 2.5 inches per cell
- **17+ images**: 2.0 inches per cell

Additionally, it respects `max_figure_width` (default: 15") and `max_figure_height` (default: 20") constraints to ensure plots render properly in notebooks.

## Format Detection

The function automatically detects and converts between different image formats:

1. **Channel Detection:** Identifies if channels are in the first (`C, H, W`) or last (`H, W, C`) dimension
2. **Batch Detection:** Handles both batched and single images
3. **Type Conversion:** Automatically converts PyTorch tensors to numpy arrays
4. **Value Normalization:** Handles both `[0, 1]` and `[0, 255]` value ranges

## Notes

- The function assumes that if the first dimension is 1 or 3 and is smaller than the other dimensions, it represents channels (`C, H, W` format)
- Images with values in `[0, 1]` are displayed directly
- Images with values > 1 are assumed to be in `[0, 255]` range and normalized to `[0, 1]`
- Grayscale images can be 2D `(H, W)` or 3D `(H, W, 1)`
