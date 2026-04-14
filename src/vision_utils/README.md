# Vision Utils

`vision_utils` provides image loading, notebook plotting, and mmap-backed image
datasets.

## Public Exports

- `read_images`
- `read_images_cpu`
- `read_images_gpu`
- `plot_images_notebook`
- `ImageMmap`
- `ImageMmapDynamic`

## Image Loading

All three image-loading helpers return a dict mapping each input path to a NumPy
array or `None` on failure.

```python
from vision_utils import read_images, read_images_cpu, read_images_gpu

paths = ["img1.jpg", "img2.png"]

auto_images = read_images(paths)
cpu_images = read_images_cpu(paths)
gpu_images = read_images_gpu(paths)

print(auto_images[paths[0]].shape)
```

### `read_images_cpu()`

- backend: Pillow
- return shape: RGB arrays in `(H, W, C)` format
- optional resize via `hw=(height, width)`

### `read_images_gpu()`

- backend: NVIDIA DALI
- return shape: RGB arrays in `(H, W, C)` format
- optional validation and resize
- returns the same dict[path, ndarray | None] structure as the CPU path

### `read_images()`

- tries GPU loading first
- falls back to CPU loading on failure
- keeps the same return type as the explicit CPU/GPU loaders

## Notebook Plotting

`plot_images_notebook()` plots arrays or tensors directly.

Supported input shapes:

- single image: `(H, W)`, `(H, W, C)`, `(C, H, W)`
- batches: `(B, H, W, C)`, `(B, C, H, W)`
- lists or tuples of arrays / tensors

If you loaded images with `read_images*()`, pass `list(images.values())`.

```python
from vision_utils import plot_images_notebook, read_images

paths = ["img1.jpg", "img2.png"]
images = read_images(paths)

plot_images_notebook(list(images.values()))
```

Current plotting defaults and behavior:

- automatic grid sizing when `nrows` and `ncols` are omitted
- automatic normalization for channel-first vs channel-last arrays
- default `dpi=300`
- adaptive figure sizing capped by `max_figure_width` and `max_figure_height`

## Mmap-Backed Datasets

### `ImageMmap`

`ImageMmap` builds or reuses a fixed-shape mmap cache from a sequence of image
paths.

```python
from vision_utils import ImageMmap

paths = ["img1.jpg", "img2.jpg"]
dataset = ImageMmap(paths, size=(224, 224))
img = dataset[0]
```

Important current behavior:

- first positional argument is a sequence of image paths
- cache files are created automatically when needed
- optional `mmap_path` lets you control the cache location

### `ImageMmapDynamic`

`ImageMmapDynamic` stores variable-shaped images in a flat mmap file plus a
metadata file.

```python
from vision_utils import ImageMmapDynamic

paths = ["img1.jpg", "img2.jpg"]
dataset = ImageMmapDynamic(paths)
img = dataset[0]
```

Use it when you want to preserve original image sizes instead of resizing to a
fixed shape.

## Practical Notes

- `plot_images_notebook()` expects arrays/tensors, not the dict returned by
  `read_images()`
- `read_images*()` returns `None` for failed paths instead of raising by default
- `ImageMmap` and `ImageMmapDynamic` build cache files under `.cache/` by
  default when you do not pass an explicit `mmap_path`
