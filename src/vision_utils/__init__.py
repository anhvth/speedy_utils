from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "plot_images_notebook",
    "read_images_cpu",
    "read_images_gpu",
    "read_images",
    "ImageMmap",
    "ImageMmapDynamic",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ImageMmap": ("vision_utils.io_utils", "ImageMmap"),
    "ImageMmapDynamic": ("vision_utils.io_utils", "ImageMmapDynamic"),
    "read_images": ("vision_utils.io_utils", "read_images"),
    "read_images_cpu": ("vision_utils.io_utils", "read_images_cpu"),
    "read_images_gpu": ("vision_utils.io_utils", "read_images_gpu"),
    "plot_images_notebook": ("vision_utils.plot", "plot_images_notebook"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_ATTRS.keys()})
