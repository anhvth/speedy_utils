"""Durable item stages and artifacts for ParallelLLMJob."""

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path


class RejectItem(Exception):
    """An unsuitable item should be replaced, without retrying it."""


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


_metrics = {}
_lock = threading.Lock()
_last_report = 0.0


def record(directory, stage, **changes):
    global _last_report
    if directory is None:
        return
    with _lock:
        for key, value in changes.items():
            name = f"{stage}.{key}"
            _metrics[name] = _metrics.get(name, 0) + value
        if time.monotonic() - _last_report >= 2:
            atomic_write(Path(directory) / f"{os.getpid()}.json",
                         json.dumps(_metrics).encode())
            _last_report = time.monotonic()


class JobContext:
    def __init__(self, root, item, semantic_config, metrics_dir=None):
        self.root = Path(root).resolve()
        identity = json.dumps([item, semantic_config], sort_keys=True, ensure_ascii=False)
        self.key = hashlib.sha256(identity.encode()).hexdigest()
        self.directory = self.root / ".stages" / self.key
        self.metrics_dir = metrics_dir
        self._history = []

    def stage(self, name, fn, *, checkpoint=True, retries=0,
              retry_if=None, retry_delay=1.0):
        if retries < 0 or retry_delay < 0:
            raise ValueError("Retry count and delay must be non-negative")
        # Include earlier results so a changed upstream stage invalidates descendants.
        key = hashlib.sha256(json.dumps([name, self._history], sort_keys=True).encode()).hexdigest()
        path = self.directory / f"{key}.json"
        if checkpoint and path.exists():
            result = json.loads(path.read_text())
            record(self.metrics_dir, name, cached=1)
        else:
            record(self.metrics_dir, name, active=1)
            started = time.monotonic()
            try:
                for attempt in range(retries + 1):
                    try:
                        result = fn()
                        if hasattr(result, "model_dump"):
                            result = result.model_dump(mode="json")
                        break
                    except RejectItem:
                        raise
                    except Exception as exc:
                        if attempt == retries or retry_if is None or not retry_if(exc):
                            raise
                        record(self.metrics_dir, name, retries=1)
                        time.sleep(min(30, retry_delay * 2 ** attempt))
                if checkpoint:
                    atomic_write(path, json.dumps(result, ensure_ascii=False).encode())
            except Exception:
                record(self.metrics_dir, name, failed=1)
                raise
            finally:
                record(self.metrics_dir, name, active=-1, calls=1,
                       seconds=time.monotonic() - started)
        self._history.append([name, result])
        return result

    def save_artifact(self, name, data):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact name must be relative and stay inside the run")
        path = self.root / relative.parent / f"{self.key}-{relative.name}"
        atomic_write(path, data)
        return path.relative_to(self.root).as_posix()

    def reject(self, category, reason):
        raise RejectItem(f"{category}: {reason}")


def monitor(stop, state_path, metrics_dir, interval):
    from tqdm import tqdm

    previous = json.loads(state_path.read_text()).get("written_rows", 0) if state_path.exists() else 0
    last = time.monotonic()
    while not stop.wait(interval):
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        now = time.monotonic()
        written = state.get("written_rows", 0)
        rate = (written - previous) / (now - last)
        remaining = (state.get("target_rows") or written) - written
        eta = f"{remaining / rate / 3600:.1f}h" if rate > 0 else "waiting"
        previous, last = written, now
        totals = {}
        for path in metrics_dir.glob("*.json"):
            for key, value in json.loads(path.read_text()).items():
                totals[key] = totals.get(key, 0) + value
        stages = sorted({key.split(".")[0] for key in totals})
        details = []
        for stage in stages:
            mean = totals.get(stage + ".seconds", 0) / max(1, totals.get(stage + ".calls", 0))
            details.append(f"{stage}:active~{totals.get(stage + '.active', 0)} "
                           f"mean={mean:.1f}s retries={totals.get(stage + '.retries', 0)} "
                           f"failed={totals.get(stage + '.failed', 0)}")
        tqdm.write(f"STATUS committed={state.get('written_rows', 0)}/{state.get('target_rows')} "
                   f"rate={rate:.2f}/s ETA={eta} "
                   f"failed={state.get('failed', 0)} rejected={state.get('rejected', 0)} | "
                   + " | ".join(details))
        import sys
        sys.stdout.flush()
