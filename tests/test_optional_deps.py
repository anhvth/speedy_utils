# tests/test_optional_deps.py
"""Optional extras must fail with a `pip install speedy-utils[extra]` hint.

Blocking a module with `sys.modules[name] = None` is how an uninstalled
dependency behaves: every `import name` inside a function raises
ModuleNotFoundError. The contract is that our guard turns that into a plain
ImportError that names the extra.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGES = ["speedy_utils", "llm_utils", "vision_utils", "datasets_utils"]
EXTRAS = {
    "matplotlib": "plot",
    "pandas": "pandas",
    "datasets": "datasets",
}


def _block(monkeypatch, *modules):
    """Make `import <module>` (and its submodules) fail like a missing package."""
    for name in list(sys.modules):
        if any(name == m or name.startswith(m + ".") for m in modules):
            monkeypatch.setitem(sys.modules, name, None)
    for module in modules:
        monkeypatch.setitem(sys.modules, module, None)


def _assert_extra_hint(exc, extra):
    assert not isinstance(exc, ModuleNotFoundError), (
        f"missing {extra} leaked a bare ModuleNotFoundError: {exc}"
    )
    assert re.search(rf"pip install '?speedy-utils\[{extra}\]'?", str(exc)), exc


@pytest.mark.parametrize("module", sorted(EXTRAS))
def test_packages_import_without_optional_dependency(module):
    """`import` of every package must work when an optional dep is absent."""
    code = (
        "import sys\n"
        f"sys.modules[{module!r}] = None\n"
        f"import {', '.join(PACKAGES)}\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_plot_images_notebook_needs_plot_extra(monkeypatch):
    import numpy as np

    from vision_utils import plot_images_notebook

    _block(monkeypatch, "matplotlib")
    with pytest.raises(ImportError) as excinfo:
        plot_images_notebook([np.zeros((4, 4, 3), dtype=np.uint8)])
    _assert_extra_hint(excinfo.value, EXTRAS["matplotlib"])


def test_multi_thread_dataframe_needs_pandas_extra(monkeypatch):
    pd = pytest.importorskip("pandas")
    from speedy_utils.multi_worker.thread import multi_thread

    df = pd.DataFrame({"a": [1, 2, 3]})
    _block(monkeypatch, "pandas")
    with pytest.raises(ImportError) as excinfo:
        multi_thread(lambda x: x, df, workers=1, progress=False)
    _assert_extra_hint(excinfo.value, EXTRAS["pandas"])


def test_multi_thread_plain_inputs_without_pandas(monkeypatch):
    """Guarding pandas must not make pandas mandatory for every call."""
    from speedy_utils.multi_worker.thread import multi_thread

    _block(monkeypatch, "pandas")
    assert multi_thread(lambda x: x * 2, [1, 2, 3], workers=1, progress=False) == [
        2,
        4,
        6,
    ]


def test_hf_dataset_row_source_needs_datasets_extra(monkeypatch):
    from datasets_utils.pcat._shared import HFDatasetRowSource

    _block(monkeypatch, "datasets")
    with pytest.raises(ImportError) as excinfo:
        HFDatasetRowSource.from_path(Path("/nonexistent/hf_dataset"))
    _assert_extra_hint(excinfo.value, EXTRAS["datasets"])


def test_merge_shards_needs_datasets_extra(monkeypatch):
    from speedy_utils.multi_worker.dataset_sharding import _merge_shards

    _block(monkeypatch, "datasets")
    with pytest.raises(ImportError) as excinfo:
        _merge_shards([], "/tmp/speedy_optional_deps_shards")
    _assert_extra_hint(excinfo.value, EXTRAS["datasets"])


def test_identify_and_memoize_work_without_pandas(monkeypatch):
    """A plain install has no pandas; identify() must not reach for it.

    Guards the regression where `identify` did isinstance(obj, pd.DataFrame) on a
    lazy proxy whose module is None: every identify() call raised
    AttributeError: 'NoneType' object has no attribute 'DataFrame'.
    """
    from speedy_utils import identify, memoize

    _block(monkeypatch, "pandas")

    assert identify({"a": 1}) == identify({"a": 1})
    assert identify([1, 2, 3]) != identify([1, 2, 4])

    calls = []

    # cache_type='mem': a disk entry from an earlier run would hide the call
    @memoize(cache_type='mem')
    def double(x):
        calls.append(x)
        return x * 2

    assert double(3) == 6
    assert double(3) == 6
    assert calls == [3], "memoized result was not cached"
