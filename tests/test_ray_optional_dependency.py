import builtins
import importlib
import sys

import pytest

from speedy_utils.multi_worker.process import multi_process


def _block_ray_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray" or name.startswith("ray."):
            raise ImportError("No module named 'ray'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_multi_process_ray_backend_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_ray_import(monkeypatch)

    with pytest.raises(ImportError, match=r"speedy-utils\[ray\]"):
        multi_process(lambda x: x, [1, 2, 3], backend="ray", progress=False)


def test_dataset_ray_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from speedy_utils.multi_worker.dataset_ray import multi_process_dataset_ray

    _block_ray_import(monkeypatch)

    with pytest.raises(ImportError, match=r"speedy-utils\[ray\]"):
        multi_process_dataset_ray(lambda x, **_: x, [1, 2, 3], progress=False)


def test_parallel_gpu_pool_import_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_ray_import(monkeypatch)

    sys.modules.pop("speedy_utils.multi_worker.parallel_gpu_pool", None)
    with pytest.raises(ImportError, match=r"speedy-utils\[ray\]"):
        importlib.import_module("speedy_utils.multi_worker.parallel_gpu_pool")
