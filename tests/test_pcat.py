from __future__ import annotations

from datasets_utils.pcat._shared import RowView, all_container_paths, all_scalar_paths, smart_expand


def test_smart_expand_expands_full_multiline_scalars_for_large_rows() -> None:
    value = {
        "title": "example",
        "payload": {
            "body": "line\n" * 200,
            "notes": ["short", "another\nmultiline\nvalue"],
        },
    }

    view = RowView(value=value)

    smart_expand(view, screen_height=5)

    assert view.expanded == set(all_container_paths(value))
    assert view.scalar_expanded == set(all_scalar_paths(value))
