from pathlib import Path

from speedy_utils import MirrowPath


def test_mirrow_path_returns_ready_mirror(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights.bin").write_bytes(b"weights")

    mirror = MirrowPath(source, mirror_root=tmp_path / "mirror", workers=2)

    assert isinstance(mirror, Path)
    assert mirror == tmp_path / "mirror" / "model"
    assert mirror.source == source.resolve()
    assert (mirror / "config.json").read_text() == "{}"
    assert (mirror / "weights.bin").read_bytes() == b"weights"
    assert not (mirror / "config.json").is_symlink()


def test_mirrow_path_reuses_current_files(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    file = source / "config.json"
    file.write_text("{}")

    first = MirrowPath(source, mirror_root=tmp_path / "mirror")
    second = MirrowPath(source, mirror_root=tmp_path / "mirror")

    assert first == second
    assert second.joinpath("config.json").read_text() == "{}"

