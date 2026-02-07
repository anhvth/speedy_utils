from pathlib import Path

from speedy_utils.scripts.prefetch_large_model import DEFAULT_EXTS, prefetch_file, scan


def test_scan_filters_extensions_and_counts_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"x" * 3)
    (tmp_path / "b.json").write_bytes(b"{}")
    (tmp_path / "c.txt").write_bytes(b"nope")

    files, total = scan(tmp_path, DEFAULT_EXTS)
    assert {p.name for p in files} == {"a.safetensors", "b.json"}
    assert total == 5


def test_prefetch_file_reads_entire_file(tmp_path: Path) -> None:
    path = tmp_path / "x.safetensors"
    content = b"hello world" * 100
    path.write_bytes(content)

    read_bytes, errors = prefetch_file(path, chunk_bytes=7)
    assert errors == 0
    assert read_bytes == len(content)


def test_prefetch_file_returns_error_on_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.safetensors"
    read_bytes, errors = prefetch_file(missing, chunk_bytes=16)
    assert read_bytes == 0
    assert errors == 1

