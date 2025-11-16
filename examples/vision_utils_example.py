"""
Example usage of vision_utils.plot_images_notebook
"""

import numpy as np

from vision_utils import plot_images_notebook


def test_auto_grid():
    """Test auto grid layout with sqrt calculation."""
    print("Testing auto grid (sqrt) with 9 images...")
    images = np.random.rand(9, 64, 64, 3)
    plot_images_notebook(images)  # Should create 3x3 grid


def test_auto_grid_non_square():
    """Test auto grid with non-perfect square number of images."""
    print("Testing auto grid with 8 images...")
    images = np.random.rand(8, 64, 64, 3)
    plot_images_notebook(images)  # Should create 3x3 grid (with 1 empty)


def test_manual_grid():
    """Test with manual grid specification."""
    print("Testing manual 2x4 grid...")
    images = np.random.rand(8, 64, 64, 3)
    plot_images_notebook(images, nrows=2, ncols=4)


def test_many_images():
    """Test with many images (adaptive sizing)."""
    print("Testing with 25 images (adaptive sizing)...")
    images = np.random.rand(25, 64, 64, 3)
    plot_images_notebook(images)  # Should create 5x5 grid with smaller cells


def test_numpy_bhwc():
    """Test with numpy array in (B, H, W, C) format."""
    print("Testing numpy array (B, H, W, C) format...")
    images = np.random.rand(8, 64, 64, 3)
    plot_images_notebook(images)


def test_numpy_bchw():
    """Test with numpy array in (B, C, H, W) format."""
    print("Testing numpy array (B, C, H, W) format...")
    images = np.random.rand(8, 3, 64, 64)
    plot_images_notebook(images)


def test_list_of_arrays():
    """Test with list of numpy arrays in different formats."""
    print("Testing list of numpy arrays...")
    images = [
        np.random.rand(64, 64, 3),  # (H, W, C)
        np.random.rand(3, 64, 64),  # (C, H, W)
        np.random.rand(64, 64),  # Grayscale (H, W)
        np.random.rand(64, 64, 1),  # Grayscale (H, W, 1)
    ]
    plot_images_notebook(images, titles=["HWC", "CHW", "Gray", "Gray1"])


def test_torch_tensor():
    """Test with PyTorch tensor."""
    try:
        import torch

        print("Testing PyTorch tensor (B, C, H, W) format...")
        images = torch.rand(8, 3, 64, 64)
        plot_images_notebook(images)
    except ImportError:
        print("PyTorch not installed, skipping torch test")


def test_single_image():
    """Test with single image."""
    print("Testing single image...")
    image = np.random.rand(128, 128, 3)
    plot_images_notebook(image)


def test_custom_dpi():
    """Test with custom DPI for high resolution."""
    print("Testing custom DPI...")
    images = np.random.rand(4, 64, 64, 3)
    plot_images_notebook(images, dpi=100)


if __name__ == "__main__":
    # Run examples
    print("=== Auto Grid Tests ===")
    test_auto_grid()
    test_auto_grid_non_square()

    print("\n=== Manual Grid Test ===")
    test_manual_grid()

    print("\n=== Adaptive Sizing Test ===")
    test_many_images()

    print("\n=== Format Tests ===")
    test_numpy_bhwc()
    test_numpy_bchw()
    test_list_of_arrays()
    test_torch_tensor()

    print("\n=== Edge Cases ===")
    test_single_image()
    test_custom_dpi()
