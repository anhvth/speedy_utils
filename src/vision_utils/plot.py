from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False


def _to_numpy(img: Any) -> np.ndarray:
    """Convert image to numpy array."""
    if TORCH_AVAILABLE and torch is not None and isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError(
            f"Unsupported image type: {type(img)}. "
            "Expected numpy.ndarray or torch.Tensor"
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
    elif img.ndim == 3:
        # Check if it's (C, H, W) format
        if img.shape[0] in [1, 3] and img.shape[0] < min(img.shape[1:]):
            # Likely (C, H, W) -> transpose to (H, W, C)
            return np.transpose(img, (1, 2, 0))
        else:
            # Already (H, W, C)
            return img
    else:
        raise ValueError(
            f"Invalid image shape: {img.shape}. " "Expected 2D or 3D array"
        )


def _normalize_batch(
    images: Union[np.ndarray, List[np.ndarray], List[Any], Any],
) -> List[np.ndarray]:
    """
    Normalize batch of images to list of (H, W, C) numpy arrays.

    Handles:
    - List of numpy arrays or torch tensors
    - Single numpy array of shape (B, H, W, C) or (B, C, H, W)
    - Single torch tensor of shape (B, H, W, C) or (B, C, H, W)
    """
    # Convert to numpy if torch tensor
    if TORCH_AVAILABLE and torch is not None and isinstance(images, torch.Tensor):
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
                f"Invalid array shape: {images.shape}. " "Expected 2D, 3D, or 4D array"
            )

    # Handle list of images
    if isinstance(images, list):
        normalized = []
        for img in images:
            img_np = _to_numpy(img)
            img_normalized = _normalize_image_format(img_np)
            normalized.append(img_normalized)
        return normalized

    raise TypeError(
        f"Unsupported images type: {type(images)}. "
        "Expected list, numpy.ndarray, or torch.Tensor"
    )


def plot_images_notebook(
    images: Union[np.ndarray, List[np.ndarray], List[Any], Any],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    titles: Optional[List[str]] = None,
    cmap: Optional[str] = None,
    dpi: int = 72,
    max_figure_width: float = 15.0,
    max_figure_height: float = 20.0,
):
    """
    Plot a batch of images in a notebook with smart grid layout.

    Args:
        images: Images to plot. Can be:
            - List of numpy arrays or torch tensors
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

        >>> # List of images with different formats
        >>> images = [
        ...     np.random.rand(64, 64, 3),  # (H, W, C)
        ...     np.random.rand(3, 64, 64),  # (C, H, W)
        ...     torch.rand(64, 64),         # Grayscale
        ... ]
        >>> plot_images_notebook(images, ncols=3)
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        raise ImportError(
            "matplotlib is required for plot_images_notebook. "
            "Install it with: pip install matplotlib"
        )

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
        # Calculate based on image aspect ratio
        sample_img = images_list[0]
        img_height, img_width = sample_img.shape[:2]
        aspect_ratio = img_width / img_height

        # Target cell size in inches (smaller for many images)
        if n_images <= 4:
            cell_width = 4.0
        elif n_images <= 9:
            cell_width = 3.0
        elif n_images <= 16:
            cell_width = 2.5
        else:
            cell_width = 2.0

        cell_height = cell_width / aspect_ratio

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

    for idx, (ax, img) in enumerate(zip(axes_flat, images_list)):
        # Determine if grayscale
        is_grayscale = img.shape[-1] == 1

        if is_grayscale:
            ax.imshow(img[:, :, 0], cmap=cmap or "gray")
        else:
            # Clip values to [0, 1] if they look like normalized images
            if img.max() <= 1.0:
                img_display = np.clip(img, 0, 1)
            else:
                # Assume [0, 255] range
                img_display = np.clip(img / 255.0, 0, 1)
            ax.imshow(img_display)

        ax.axis("off")

        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=8 if n_images > 9 else 10)

    # Hide unused subplots
    for idx in range(n_images, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.show()
