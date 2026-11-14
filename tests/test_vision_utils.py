"""Regression tests for vision_utils plotting and mmap caches."""

from __future__ import annotations

import numpy as np
import pytest


def _write_image(path, color):
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8)).save(path)
    return str(path)


class _FakeAxes:
    def __init__(self):
        self.images = []

    def imshow(self, data, **kwargs):
        self.images.append(data)

    def axis(self, *args, **kwargs):
        pass

    def set_title(self, *args, **kwargs):
        pass


class _FakePyplot:
    def __init__(self):
        self.axes = []

    def subplots(self, nrows, ncols, **kwargs):
        grid = [[_FakeAxes() for _ in range(ncols)] for _ in range(nrows)]
        self.axes = [ax for row in grid for ax in row]
        return object(), np.array(grid, dtype=object)

    def tight_layout(self):
        pass

    def show(self):
        pass


def test_plot_images_notebook_renders_image_paths(tmp_path, monkeypatch):
    """A list of image paths must plot the images, not the path strings."""
    from vision_utils import plot

    paths = [
        _write_image(tmp_path / "red.png", (255, 0, 0)),
        _write_image(tmp_path / "blue.png", (0, 0, 255)),
    ]
    fake_plt = _FakePyplot()
    monkeypatch.setattr(plot, "_check_matplotlib_available", lambda: (True, fake_plt))

    plot.plot_images_notebook(paths)

    rendered = fake_plt.axes[0].images[0]
    assert rendered.shape == (8, 8, 3)
    # uint8 images are scaled to [0, 1] before imshow
    assert rendered[0, 0].tolist() == [1.0, 0.0, 0.0]


def test_imagemmap_builds_cache_and_keeps_lock_file(tmp_path):
    """The first build writes the cache; the lock file must stay on disk."""
    from vision_utils.io_utils import ImageMmap

    path = _write_image(tmp_path / "a.png", (255, 0, 0))
    cache = tmp_path / "cache.dat"

    dataset = ImageMmap([path], size=(8, 8), mmap_path=cache)

    assert dataset[0][0, 0].tolist() == [255, 0, 0]
    assert cache.exists()
    assert (tmp_path / "cache.dat.lock").exists()


def test_imagemmap_reopen_reuses_cache_without_rebuilding(tmp_path, monkeypatch):
    """An unchanged image list must be served from the existing cache."""
    from vision_utils.io_utils import ImageMmap

    paths = [
        _write_image(tmp_path / "a.png", (255, 0, 0)),
        _write_image(tmp_path / "b.png", (0, 255, 0)),
    ]
    cache = tmp_path / "cache.dat"
    ImageMmap(paths, size=(8, 8), mmap_path=cache)

    rebuilds = []
    monkeypatch.setattr(ImageMmap, "_build_cache", lambda *args, **kwargs: rebuilds.append(args))
    reopened = ImageMmap(paths, size=(8, 8), mmap_path=cache)

    assert rebuilds == []
    assert reopened[1][0, 0].tolist() == [0, 255, 0]


def test_imagemmap_rebuilds_when_image_list_changes(tmp_path):
    """A changed image list invalidates the hash and rebuilds the cache."""
    from vision_utils.io_utils import ImageMmap

    first = _write_image(tmp_path / "a.png", (255, 0, 0))
    second = _write_image(tmp_path / "b.png", (0, 0, 255))
    cache = tmp_path / "cache.dat"
    ImageMmap([first], size=(8, 8), mmap_path=cache)

    rebuilt = ImageMmap([first, second], size=(8, 8), mmap_path=cache)

    assert rebuilt.n == 2
    assert rebuilt[1][0, 0].tolist() == [0, 0, 255]


def test_imagemmap_dynamic_rebuilds_when_image_list_changes(tmp_path):
    """Dynamic mmap rebuilds on hash change and keeps original image shapes."""
    from vision_utils.io_utils import ImageMmapDynamic

    first = _write_image(tmp_path / "a.png", (255, 0, 0))
    second = _write_image(tmp_path / "b.png", (0, 0, 255))
    cache = tmp_path / "dynamic.dat"

    dataset = ImageMmapDynamic([first], mmap_path=cache)
    assert dataset[0].shape == (8, 8, 3)

    rebuilt = ImageMmapDynamic([first, second], mmap_path=cache)
    assert rebuilt.n == 2
    assert rebuilt[1][0, 0].tolist() == [0, 0, 255]
