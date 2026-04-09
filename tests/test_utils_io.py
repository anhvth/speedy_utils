from __future__ import annotations

import json

from speedy_utils import load_by_ext


def test_load_by_ext_expands_home_directory(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    data_file = home_dir / "sample.json"
    data_file.write_text(json.dumps({"value": 42}), encoding="utf-8")

    monkeypatch.setenv("HOME", str(home_dir))

    result = load_by_ext("~/sample.json")

    assert result == {"value": 42}
