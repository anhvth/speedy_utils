"""Spawn/callable support for the multiprocessing backend."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


_MISSING = object()


class _SpawnFallbackRequired(RuntimeError):
    """Internal signal that mp spawn payload is not serializable."""

    def __init__(self, *, reason: str, exc: Exception) -> None:
        super().__init__(f"{reason}: {type(exc).__name__}: {exc}")
        self.reason = reason
        self.exc = exc


def _resolve_attr_path(root: Any, attr_path: str) -> Any:
    current = root
    for part in attr_path.split("."):
        current = getattr(current, part)
    return current


def _is_spawn_importable_callable(func: Callable[[Any], Any]) -> bool:
    """Return True when ``spawn`` can re-import this callable by name."""
    module_name = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not module_name or not qualname:
        return False
    if module_name == "__main__" or "<locals>" in qualname:
        return False

    try:
        module = importlib.import_module(module_name)
        resolved = _resolve_attr_path(module, qualname)
    except (AttributeError, ImportError):
        return False
    return resolved is func


def _patch_main_spec_for_spawn() -> tuple[Any, Any]:
    """Temporarily give ``__main__`` a real ``__spec__`` before spawning."""
    main = sys.modules.get("__main__")
    if main is None:
        return None, _MISSING
    original_spec = main.__dict__.get("__spec__", _MISSING)
    if original_spec is not _MISSING and original_spec is not None:
        return None, _MISSING

    our_mod = sys.modules.get(__name__)
    our_spec = getattr(our_mod, "__spec__", None) if our_mod else None
    if our_spec is None:
        return None, _MISSING

    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*found in sys\.modules after import.*",
        category=RuntimeWarning,
        module=r"runpy",
    )
    main.__spec__ = our_spec
    return main, original_spec


def _restore_main_spec_for_spawn(state: tuple[Any, Any]) -> None:
    """Undo the temporary ``__spec__`` patch set for spawn."""
    main, original_spec = state
    if main is None:
        return
    if original_spec is _MISSING:
        main.__dict__.pop("__spec__", None)
    else:
        main.__spec__ = original_spec


@contextmanager
def patched_spawn_environment() -> Iterator[None]:
    """Apply and restore the environment hacks needed for safe spawn."""
    spec_patch_state = _patch_main_spec_for_spawn()
    prev_child_flag = os.environ.get("_SPEEDY_MP_CHILD")
    prev_pywarn = os.environ.get("PYTHONWARNINGS", "")
    warn_filter = "ignore::RuntimeWarning:runpy"

    os.environ["_SPEEDY_MP_CHILD"] = "1"
    if warn_filter not in prev_pywarn:
        os.environ["PYTHONWARNINGS"] = f"{prev_pywarn},{warn_filter}".lstrip(",")

    try:
        yield
    finally:
        if prev_child_flag is None:
            os.environ.pop("_SPEEDY_MP_CHILD", None)
        else:
            os.environ["_SPEEDY_MP_CHILD"] = prev_child_flag

        if prev_pywarn:
            os.environ["PYTHONWARNINGS"] = prev_pywarn
        else:
            os.environ.pop("PYTHONWARNINGS", None)

        _restore_main_spec_for_spawn(spec_patch_state)


def _infer_importable_module(func: Callable[[Any], Any]) -> tuple[str, str] | None:
    """For a ``__main__`` function, discover its real importable module."""
    qualname = getattr(func, "__qualname__", None)
    if not qualname or "<locals>" in qualname:
        return None

    try:
        source_file = inspect.getfile(func)
    except (TypeError, OSError):
        return None

    source_path = Path(source_file).resolve()
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    script_dir: Path | None = None
    if main_file and getattr(main_module, "__spec__", None) is None:
        script_dir = Path(main_file).resolve().parent

    for sp in sorted(sys.path, key=len, reverse=True):
        if not sp:
            continue
        sp_path = Path(sp).resolve()
        if script_dir is not None and sp_path == script_dir:
            continue
        try:
            rel = source_path.relative_to(sp_path)
        except ValueError:
            continue

        parts = list(rel.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts = parts[:-1]

        module_name = ".".join(parts)
        if not module_name or module_name == "__main__":
            continue

        try:
            mod = importlib.import_module(module_name)
            resolved = _resolve_attr_path(mod, qualname)
        except (ImportError, AttributeError):
            continue

        if resolved is func:
            return module_name, qualname

        resolved_code = getattr(resolved, "__code__", None)
        func_code = getattr(func, "__code__", None)
        if (
            resolved_code is not None
            and func_code is not None
            and resolved_code == func_code
        ):
            return module_name, qualname

    return None


def _ensure_module_globals(module_name: str | None, script_path: str | None) -> None:
    """Populate module globals by re-importing the source in the child process."""
    if module_name == "__main__" and script_path:
        import runpy

        mod_dict = runpy.run_path(script_path, run_name="__mp_child__")
        skip = {
            "__builtins__",
            "__name__",
            "__file__",
            "__doc__",
            "__spec__",
            "__loader__",
            "__package__",
            "__cached__",
            "__annotations__",
        }
        target = sys.modules["__main__"].__dict__
        for key, value in mod_dict.items():
            if key not in skip:
                target[key] = value
        return

    if module_name and module_name != "__main__":
        importlib.import_module(module_name)


def _serialize_spawn_callable(func: Callable[[Any], Any]) -> bytes:
    """Serialize non-importable callables for the notebook fallback path."""
    import pickle

    importable = _infer_importable_module(func)
    if importable is not None:
        return pickle.dumps(
            {
                "_v": 1,
                "import_ref": True,
                "module_name": importable[0],
                "qualname": importable[1],
            }
        )

    try:
        import dill
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "multi_process(..., backend='mp') needs 'dill' when the callable "
            "is defined in __main__, locally, or otherwise cannot be imported "
            "by child processes started with 'spawn'."
        ) from exc

    try:
        func_bytes = dill.dumps(func, recurse=True)
        return pickle.dumps({"_v": 1, "shallow": False, "func_bytes": func_bytes})
    except (TypeError, pickle.PicklingError):
        pass

    func_bytes = dill.dumps(func, recurse=False)
    module_name = getattr(func, "__module__", None)
    script_path = None
    if module_name == "__main__":
        main_mod = sys.modules.get("__main__")
        if main_mod and hasattr(main_mod, "__file__"):
            script_path = main_mod.__file__

    return pickle.dumps(
        {
            "_v": 1,
            "shallow": True,
            "func_bytes": func_bytes,
            "module_name": module_name,
            "script_path": script_path,
        }
    )


def _deserialize_spawn_callable(payload: bytes) -> Callable[[Any], Any]:
    """Restore a callable serialized for the notebook fallback path."""
    import pickle

    import dill

    try:
        data = pickle.loads(payload)
    except Exception:
        return dill.loads(payload)

    if not isinstance(data, dict) or data.get("_v") != 1:
        return dill.loads(payload)

    if data.get("import_ref"):
        mod = importlib.import_module(data["module_name"])
        return _resolve_attr_path(mod, data["qualname"])

    if data.get("shallow"):
        _ensure_module_globals(data.get("module_name"), data.get("script_path"))

    try:
        return dill.loads(data["func_bytes"])
    except RecursionError:
        raise RuntimeError(
            "dill deserialization hit infinite recursion (likely a pydantic "
            "model with `from __future__ import annotations`). Run via "
            "`python -m <module>` or use the tools/ wrapper so the worker "
            "function is importable by reference."
        ) from None


def _serialize_exception_frames(exc: Exception) -> list[tuple[str, int, str, dict]]:
    """Extract a picklable traceback payload for cross-process reporting."""
    import traceback

    tb = exc.__traceback__
    if tb is None:
        return []
    return [
        (frame.filename, frame.lineno or 0, frame.name, {})
        for frame in traceback.extract_tb(tb)
    ]


def _probe_spawn_picklable(obj: Any) -> None:
    """Raise when an object cannot be pickled for multiprocessing spawn."""
    from multiprocessing.reduction import ForkingPickler

    ForkingPickler.dumps(obj)


def prepare_spawn_callable(
    func: Callable[[Any], Any],
) -> tuple[Callable[[Any], Any] | None, bytes | None]:
    """Return either the importable callable or a serialized fallback payload."""
    if not _is_spawn_importable_callable(func):
        try:
            return None, _serialize_spawn_callable(func)
        except Exception as exc:
            raise _SpawnFallbackRequired(
                reason="callable serialization failed",
                exc=exc,
            ) from exc

    try:
        _probe_spawn_picklable(func)
    except Exception as exc:
        raise _SpawnFallbackRequired(
            reason="callable is not spawn-picklable",
            exc=exc,
        ) from exc

    return func, None


def validate_spawn_kwargs(func_kwargs: dict[str, Any]) -> None:
    """Raise when func kwargs are not serializable for spawn."""
    try:
        _probe_spawn_picklable(func_kwargs)
    except Exception as exc:
        raise _SpawnFallbackRequired(
            reason="func kwargs are not spawn-picklable",
            exc=exc,
        ) from exc
