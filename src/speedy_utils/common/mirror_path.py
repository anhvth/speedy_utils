"""Path-like model mirroring into local scratch storage."""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path


_PathType = type(Path())


def _format_bytes(value: float) -> str:
    if value >= 1 << 30:
        return f"{value / (1 << 30):.1f} GB"
    if value >= 1 << 20:
        return f"{value / (1 << 20):.0f} MB"
    return f"{value / (1 << 10):.0f} KB"


class MirrowPath(_PathType):
    """A ``Path`` whose value is a ready local mirror of the source path.

    ``MirrowPath("~/hf-ckpt/Qwen3-Omni-30B-A3B-Instruct")`` resolves the
    source and synchronously makes ``/tmp/scratch/models/Qwen3-Omni-30B-A3B-Instruct``
    available.  The returned object is the mirror path, so it can be passed
    directly to a model loader.

    Set ``MIRROW_BASE`` to change the mirror root and ``MIRROW_WORKERS`` to
    change the number of concurrent file copies.
    """

    def __new__(
        cls,
        path: os.PathLike[str] | str,
        *,
        mirror_root: os.PathLike[str] | str | None = None,
        workers: int | None = None,
    ) -> "MirrowPath":
        source = Path(path).expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"model path is not a directory: {source}")

        root = Path(
            mirror_root
            if mirror_root is not None
            else os.environ.get("MIRROW_BASE", "/tmp/scratch/models")
        ).expanduser()
        target = root / source.name

        result = super().__new__(cls, target)
        result._source = source
        result._mirror_root = root
        result._mirror_target = target
        result._mirror_workers = max(
            1,
            workers
            if workers is not None
            else int(os.environ.get("MIRROW_WORKERS", "32")),
        )
        return result

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        mirror_root: os.PathLike[str] | str | None = None,
        workers: int | None = None,
    ) -> None:
        # Path's state is initialized between __new__ and __init__ on Python
        # 3.13, so filesystem work must happen here rather than in __new__.
        super().__init__(self._mirror_target)
        self._ensure_mirror()

    def with_segments(self, *pathsegments: os.PathLike[str] | str) -> Path:
        """Keep child paths ordinary ``Path`` objects.

        This prevents ``mirror / "config.json"`` from starting another mirror
        operation while retaining all normal pathlib operators.
        """

        return Path(*pathsegments)

    @property
    def source(self) -> Path:
        """The resolved source model directory."""

        return self._source

    @property
    def mirror_path(self) -> Path:
        """The mirrored path represented by this object."""

        return _PathType(self)

    def __enter__(self) -> "MirrowPath":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    @staticmethod
    def _copy_file(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        try:
            shutil.copy2(source, partial)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)

    @staticmethod
    def _is_current(source: Path, target: Path) -> bool:
        if target.is_symlink() or not target.is_file():
            return False
        source_stat = source.stat()
        target_stat = target.stat()
        return (
            target_stat.st_size == source_stat.st_size
            and target_stat.st_mtime_ns == source_stat.st_mtime_ns
        )

    def _ensure_mirror(self) -> None:
        source = self._source
        target = _PathType(self)
        if source == target.resolve():
            return

        target.mkdir(parents=True, exist_ok=True)
        files = [file for file in source.rglob("*") if file.is_file()]
        total_bytes = sum(file.stat().st_size for file in files)
        print(
            f"[MirrowPath] copy: {len(files)} files ({_format_bytes(total_bytes)}) "
            f"from {source} to {target} with {self._mirror_workers} workers"
        )

        jobs: dict[Future[None], tuple[Path, int]] = {}
        skipped = 0
        copied = 0
        copied_bytes = 0
        errors: list[tuple[Path, Exception]] = []
        started = time.monotonic()

        with ThreadPoolExecutor(max_workers=self._mirror_workers) as pool:
            for file in files:
                destination = target / file.relative_to(source)
                if self._is_current(file, destination):
                    skipped += 1
                    continue
                jobs[pool.submit(self._copy_file, file, destination)] = (
                    file,
                    file.stat().st_size,
                )

            for job in as_completed(jobs):
                file, size = jobs[job]
                try:
                    job.result()
                    copied += 1
                    copied_bytes += size
                    done = copied + skipped
                    if done == 1 or done % 5 == 0 or done == len(files):
                        elapsed = time.monotonic() - started
                        rate = copied_bytes / elapsed if elapsed else 0
                        print(
                            f"[MirrowPath] copy: {done}/{len(files)}  "
                            f"{_format_bytes(copied_bytes)}  {_format_bytes(rate)}/s"
                        )
                except Exception as exc:  # pragma: no cover - filesystem-specific
                    errors.append((file, exc))

        elapsed = time.monotonic() - started
        rate = copied_bytes / elapsed if elapsed else 0
        print(
            f"[MirrowPath] copy done: copied={copied} ({_format_bytes(copied_bytes)}), "
            f"skipped={skipped}, errors={len(errors)}, "
            f"time={elapsed:.1f}s ({_format_bytes(rate)}/s)"
        )
        if errors:
            details = "; ".join(f"{file}: {exc}" for file, exc in errors[:3])
            raise RuntimeError(f"mirror incomplete: {details}")

        incomplete = [
            file
            for file in files
            if not self._is_current(file, target / file.relative_to(source))
        ]
        if incomplete:
            raise RuntimeError(
                f"mirror incomplete: {len(incomplete)} file(s) missing or mismatched"
            )
