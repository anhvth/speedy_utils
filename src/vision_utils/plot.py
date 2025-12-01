from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


if TYPE_CHECKING:

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from .io_utils import read_images


def _check_torch_available():
    """Check if torch is available without importing at module level."""
    try:
        import torch

        return True, torch
    except ImportError:
        return False, None


def _check_matplotlib_available():
    """Check if matplotlib is available without importing at module level."""
    try:
        import matplotlib.pyplot as plt

        return True, plt
    except ImportError:
        return False, None


def _to_numpy(img: Any) -> np.ndarray:
    """Convert image to numpy array."""
    torch_available, torch = _check_torch_available()
    if torch_available and torch is not None and isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy()
    if isinstance(img, np.ndarray):
        return img
    raise TypeError(
        f'Unsupported image type: {type(img)}. Expected numpy.ndarray or torch.Tensor'
    )


def _normalize_image_format(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to (H, W, C) format.

    Detects and converts from:
    - (C, H, W) where C is 1 or 3
    - (H, W) grayscale
    - (H, W, C) already correct
    """
    if img.ndim == 2:
        # Grayscale (H, W) -> (H, W, 1)
        return img[:, :, np.newaxis]
    if img.ndim == 3:
        # Check if it's (C, H, W) format
        if img.shape[0] in [1, 3] and img.shape[0] < min(img.shape[1:]):
            # Likely (C, H, W) -> transpose to (H, W, C)
            return np.transpose(img, (1, 2, 0))
        # Already (H, W, C)
        return img
    raise ValueError(f'Invalid image shape: {img.shape}. Expected 2D or 3D array')


def _normalize_batch(
    images: Union[np.ndarray, List[np.ndarray], List[Any], Any],
) -> List[np.ndarray]:
    """
    Normalize batch of images to list of (H, W, C) numpy arrays.

    Handles:
    - List of numpy arrays or torch tensors
    - List of file paths (strings or Path objects)
    - Single numpy array of shape (B, H, W, C) or (B, C, H, W)
    - Single torch tensor of shape (B, H, W, C) or (B, C, H, W)
    """
    # Convert to numpy if torch tensor
    torch_available, torch = _check_torch_available()
    if torch_available and torch is not None and isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # Handle single numpy array with batch dimension
    if isinstance(images, np.ndarray):
        if images.ndim == 4:
            # (B, H, W, C) or (B, C, H, W)
            # Check if it's (B, C, H, W) format
            if images.shape[1] in [1, 3] and images.shape[1] < min(images.shape[2:]):
                # (B, C, H, W) -> transpose to (B, H, W, C)
                images = np.transpose(images, (0, 2, 3, 1))
            # Convert to list of images
            images = [images[i] for i in range(images.shape[0])]
        elif images.ndim == 3:
            # Single image (H, W, C) or (C, H, W)
            images = [_normalize_image_format(images)]
        elif images.ndim == 2:
            # Single grayscale image (H, W)
            images = [images[:, :, np.newaxis]]
        else:
            raise ValueError(
                f'Invalid array shape: {images.shape}. Expected 2D, 3D, or 4D array'
            )

    # Handle list of images
    if isinstance(images, list):
        path_indices = [
            idx for idx, img in enumerate(images) if isinstance(img, (str, Path))
        ]

        # Bulk load any file paths while preserving order
        loaded_paths = {}
        if path_indices:
            loaded_arrays = read_images([str(images[idx]) for idx in path_indices])

            if len(loaded_arrays) != len(path_indices):
                raise ValueError(
                    'Number of loaded images does not match number of paths provided.'
                )

            for idx, arr in zip(path_indices, loaded_arrays, strict=False):
                loaded_paths[idx] = arr

        normalized = []
        for idx, img in enumerate(images):
            if idx in loaded_paths:
                img_np = loaded_paths[idx]
            else:
                img_np = _to_numpy(img)
            img_normalized = _normalize_image_format(img_np)
            normalized.append(img_normalized)
        return normalized

    raise TypeError(
        f'Unsupported images type: {type(images)}. '
        'Expected list, numpy.ndarray, or torch.Tensor'
    )


def plot_images_notebook(
    images: Union[np.ndarray, List[np.ndarray], List[Any], Any, Tuple],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    titles: Optional[List[str]] = None,
    cmap: Optional[str] = None,
    dpi: int = 300,
    max_figure_width: float = 15.0,
    max_figure_height: float = 20.0,
):
    """
    Plot a batch of images in a notebook with smart grid layout.
    Handles images of different shapes gracefully.

    Args:
        images: Images to plot. Can be:
            - List of numpy arrays or torch tensors (can have different shapes)
            - List of image file paths (strings or Path objects)
            - Numpy array of shape (B, H, W, C) or (B, C, H, W)
            - Torch tensor of shape (B, H, W, C) or (B, C, H, W)
            Each image can be (H, W), (H, W, C), (C, H, W) format
        nrows: Number of rows in grid. If None, auto-calculated from sqrt
        ncols: Number of columns in grid. If None, auto-calculated from sqrt
        figsize: Figure size (width, height). If None, auto-calculated
        titles: List of titles for each image
        cmap: Colormap for grayscale images (default: 'gray')
        dpi: Dots per inch for the figure (default: 72)
        max_figure_width: Maximum figure width in inches (default: 15)
        max_figure_height: Maximum figure height in inches (default: 20)

    Example:
        >>> import numpy as np
        >>> # Auto grid layout with sqrt
        >>> images = np.random.rand(9, 64, 64, 3)
        >>> plot_images_notebook(images)  # 3x3 grid

        >>> # Custom grid
        >>> images = np.random.rand(8, 64, 64, 3)
        >>> plot_images_notebook(images, nrows=2, ncols=4)

        >>> # PyTorch tensor in (B, C, H, W) format
        >>> import torch
        >>> images = torch.rand(8, 3, 64, 64)
        >>> plot_images_notebook(images)

        >>> # List of images with different formats and shapes
        >>> images = [
        ...     np.random.rand(64, 64, 3),    # (H, W, C)
        ...     np.random.rand(3, 128, 128),  # (C, H, W) - different size
        ...     torch.rand(32, 48),           # Grayscale - different size
        ...     np.random.rand(100, 200, 3),  # Different aspect ratio
        ... ]
        >>> plot_images_notebook(images, ncols=2)
    """
    if isinstance(images, tuple):
        images = list(images)
    # Check matplotlib availability
    mpl_available, plt = _check_matplotlib_available()
    if not mpl_available:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

    # Normalize all images to list of (H, W, C) numpy arrays
    images_list = _normalize_batch(images)

    n_images = len(images_list)

    # Smart grid layout calculation
    if nrows is None and ncols is None:
        # Use sqrt to get roughly square grid
        ncols_calc = int(np.ceil(np.sqrt(n_images)))
        nrows_calc = int(np.ceil(n_images / ncols_calc))
        nrows = nrows_calc
        ncols = ncols_calc
    elif nrows is None:
        # Calculate rows from columns
        assert ncols is not None
        nrows = int(np.ceil(n_images / ncols))
    elif ncols is None:
        # Calculate columns from rows
        assert nrows is not None
        ncols = int(np.ceil(n_images / nrows))

    # At this point, both nrows and ncols are guaranteed to be int
    assert nrows is not None and ncols is not None

    # Auto-calculate figure size if not provided
    if figsize is None:
        # Calculate based on average aspect ratio across all images
        avg_aspect_ratio = 0.0
        for img in images_list:
            img_height, img_width = img.shape[:2]
            avg_aspect_ratio += img_width / img_height
        avg_aspect_ratio /= n_images

        # Target cell size in inches (smaller for many images)
        if n_images <= 4:
            cell_width = 4.0
        elif n_images <= 9:
            cell_width = 3.0
        elif n_images <= 16:
            cell_width = 2.5
        else:
            cell_width = 2.0

        cell_height = cell_width / avg_aspect_ratio

        fig_width = ncols * cell_width
        fig_height = nrows * cell_height

        # Constrain to max sizes to prevent notebook breaking
        if fig_width > max_figure_width:
            scale = max_figure_width / fig_width
            fig_width = max_figure_width
            fig_height *= scale

        if fig_height > max_figure_height:
            scale = max_figure_height / fig_height
            fig_height = max_figure_height
            fig_width *= scale

        figsize = (fig_width, fig_height)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    for idx, (ax, img) in enumerate(zip(axes_flat, images_list, strict=False)):
        # Determine if grayscale
        is_grayscale = img.shape[-1] == 1

        if is_grayscale:
            ax.imshow(img[:, :, 0], cmap=cmap or 'gray', aspect='auto')
        else:
            # Clip values to [0, 1] if they look like normalized images
            if img.max() <= 1.0:
                img_display = np.clip(img, 0, 1)
            else:
                # Assume [0, 255] range
                img_display = np.clip(img / 255.0, 0, 1)
            ax.imshow(img_display, aspect='auto')

        ax.axis('off')

        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=8 if n_images > 9 else 10)

    # Hide unused subplots
    for idx in range(n_images, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_tensor(img_tensor, mode='hwc', normalize=True, max_cols=8):
    """
    Visualize a tensor as an image or grid.

    Args:
        img_tensor: torch.Tensor, shape (C,H,W), (H,W,C), or (B,C,H,W)
        mode: "hwc", "chw", or "bchw"
        normalize: scale float tensor to 0–255 uint8 for display
        max_cols: max columns when tiling a batch
    """
    # Check matplotlib availability
    mpl_available, plt = _check_matplotlib_available()
    if not mpl_available:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")
    
    if mode == 'chw':
        img_tensor = img_tensor.permute(1, 2, 0)
        imgs = [img_tensor]
    elif mode == 'bchw':
        b, c, h, w = img_tensor.shape
        imgs = [img_tensor[i].permute(1, 2, 0) for i in range(b)]
    elif mode == 'hwc':
        imgs = [img_tensor]
    else:
        raise ValueError("mode must be 'hwc', 'chw', or 'bchw'")

    # normalize each image
    processed = []
    for img in imgs:
        img = img.detach().cpu().numpy()
        if normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
        processed.append(img)

    if len(processed) == 1:
        plt.imshow(processed[0])
    else:
        cols = min(max_cols, len(processed))
        rows = int(np.ceil(len(processed) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.atleast_2d(axes)
        for ax, img in zip(axes.flat, processed, strict=False):
            ax.imshow(img)
            ax.axis('off')
        for ax in axes.flat[len(processed) :]:
            ax.axis('off')
    plt.show()
