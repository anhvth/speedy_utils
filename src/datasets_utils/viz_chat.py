#!/usr/bin/env python3
"""
Visualize chat/conversation datasets in various formats.

Supports loading from:
- HuggingFace datasets (saved with save_to_disk())
- Tokenized HuggingFace datasets (requires --tokenizer)
- JSONL files (one JSON object per line)
- JSON files (single item or list of items)
- Folders of JSON files (each file treated as one item)

Supports data formats:
- ChatML: [{"role": "user", "content": "..."}, ...]
- ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
- Messages: {"messages": [...], "tools": [...]}  (OpenAI format)
- Tokenized: {"input_ids": [...], "labels": [...]}  (requires --tokenizer)
- Canonical: {"input_messages": ..., "reasoning_turn": ...}

Interactive Controls:
- Enter: View next random item (infinite)
- g: Go to specific index (0-based)
- q: Quit

Usage:
    # Infinite random sampling (default)
    viz_chat data/my_dataset

    # JSONL file
    viz_chat data/conversations.jsonl

    # Tokenized dataset (requires tokenizer to decode)
    viz_chat data/tokenized_dataset/ --tokenizer Qwen/Qwen3-8B

    # Folder of JSON files
    viz_chat data/json_folder/

    # Single JSON file
    viz_chat data/single_item.json

    # With specific format
    viz_chat data/sharegpt.jsonl --format sharegpt

    # Show tools if present
    viz_chat data/with_tools.jsonl --show-tools
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text


# Lazy import for datasets
_dataset_module = None


def _get_dataset_module():
    """Lazy load the datasets module."""
    global _dataset_module
    if _dataset_module is None:
        from datasets import Dataset, DatasetDict, load_from_disk

        _dataset_module = types.SimpleNamespace(
            Dataset=Dataset,
            DatasetDict=DatasetDict,
            load_from_disk=load_from_disk,
        )
    return _dataset_module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize chat/conversation datasets from various sources.",
        epilog=(
            "Examples:\n"
            "  viz_chat data/my_dataset.jsonl\n"
            "  viz_chat data/hf_dataset/ --count 5\n"
            "  viz_chat data/folder/ --format sharegpt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        help="Path to data source (HF dataset dir, JSONL, JSON, or folder of JSON files).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=-1,
        help="Number of items to sample (-1 for infinite random sampling, default: -1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling items (default: 0).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split name to sample from (HF datasets only).",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "chatml", "sharegpt"],
        default="auto",
        help="Input format (auto-detect by default).",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Display tools field if present.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer to use for decoding tokenized datasets (e.g., Qwen/Qwen3-8B). "
        "Required when visualizing tokenized datasets.",
    )
    return parser.parse_args(argv)


# =============================================================================
# Data Loading Functions
# =============================================================================


ItemTuple = tuple[str | None, int, dict[str, Any]]


class _LazyHFItems(Sequence[ItemTuple]):
    """Lazy, indexable view over a HuggingFace dataset loaded from disk."""

    def __init__(self, dataset: Any):
        dataset_module = _get_dataset_module()
        self._parts: list[tuple[str | None, Any, int, int]] = []

        if isinstance(dataset, dataset_module.DatasetDict):
            offset = 0
            for split_name in dataset:
                split_dataset = dataset[split_name]
                split_len = len(split_dataset)
                self._parts.append((split_name, split_dataset, offset, split_len))
                offset += split_len
            self._total_len = offset
        else:
            dataset_len = len(dataset)
            self._parts.append((None, dataset, 0, dataset_len))
            self._total_len = dataset_len

        self._offsets = [offset for _, _, offset, _ in self._parts]

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: int | slice) -> "ItemTuple | list[ItemTuple]":  # type: ignore[override]
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]  # type: ignore[misc]

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        part_idx = bisect.bisect_right(self._offsets, index) - 1
        source_name, dataset, offset, part_len = self._parts[part_idx]
        local_index = index - offset
        if local_index < 0 or local_index >= part_len:
            raise IndexError(index)

        row = dataset[local_index]
        if not isinstance(row, Mapping):
            raise ValueError(
                f"Row {local_index} in split {source_name!r} is not a mapping."
            )
        return source_name, local_index, dict(row)


def load_data(source: str | Path) -> Sequence[ItemTuple]:
    """
    Load data from various sources into a normalized format.

    Args:
        source: Path to data source (HF dataset dir, JSONL, JSON, or folder)

    Returns:
        List of tuples: (source_name, row_index, row_dict)
    """
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    if path.is_dir():
        return _load_directory(path)

    if path.suffix == ".jsonl":
        return _load_jsonl(path)

    if path.suffix == ".json":
        return _load_json(path)

    raise ValueError(
        f"Unsupported file type: {path.suffix}. Expected .jsonl, .json, or directory."
    )


def _load_directory(path: Path) -> Sequence[ItemTuple]:
    """Load from directory - either HF dataset or folder of JSON files."""
    # Check if it's a HuggingFace dataset (saved with save_to_disk)
    # DatasetDict has dataset_dict.json, Dataset has state.json or dataset_info.json
    if (
        (path / "dataset_dict.json").exists()
        or (path / "dataset_info.json").exists()
        or (path / "state.json").exists()
    ):
        return _load_hf_dataset(path)

    # Otherwise treat as folder of JSON files
    return _load_json_folder(path)


def _load_hf_dataset(path: Path) -> Sequence[ItemTuple]:
    """Load from HuggingFace dataset saved with save_to_disk()."""
    dataset_module = _get_dataset_module()

    try:
        dataset = dataset_module.load_from_disk(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HuggingFace dataset from {path}: {exc}"
        ) from exc

    return _LazyHFItems(dataset)


def _load_jsonl(path: Path) -> list[tuple[str | None, int, dict[str, Any]]]:
    """Load from JSONL file (one JSON object per line)."""
    items: list[tuple[str | None, int, dict[str, Any]]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Line {line_num} is not a JSON object")
                items.append((None, line_num - 1, item))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc
    return items


def _load_json(path: Path) -> list[tuple[str | None, int, dict[str, Any]]]:
    """Load from JSON file (single item or list of items)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items: list[tuple[str | None, int, dict[str, Any]]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {idx} is not a JSON object")
            items.append((None, idx, item))
        return items

    if isinstance(data, dict):
        return [(None, 0, data)]

    raise ValueError(f"Expected JSON object or array, got {type(data).__name__}")


# =============================================================================
# Tokenized Dataset Support
# =============================================================================


def _is_tokenized_row(row: Mapping[str, Any]) -> bool:
    """Check if a row contains tokenized data (input_ids)."""
    return "input_ids" in row and isinstance(row.get("input_ids"), list)


def _is_tokenized_dataset(items: list[tuple[str | None, int, dict[str, Any]]]) -> bool:
    """Check if the dataset appears to be tokenized."""
    if not items:
        return False
    # Check first item
    _, _, first_row = items[0]
    return _is_tokenized_row(first_row)


_tokenizer_cache: dict[str, Any] = {}


def _get_tokenizer(tokenizer_name: str) -> Any:
    """Load and cache a tokenizer by name."""
    if tokenizer_name in _tokenizer_cache:
        return _tokenizer_cache[tokenizer_name]

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for tokenized datasets. "
            "Install with: pip install transformers"
        ) from exc

    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    _tokenizer_cache[tokenizer_name] = tokenizer
    return tokenizer


def decode_tokenized_row(
    row: dict[str, Any],
    tokenizer: Any,
    show_labels: bool = True,
) -> dict[str, Any]:
    """
    Decode a tokenized row into human-readable text.

    Args:
        row: Row with input_ids, attention_mask, labels, etc.
        tokenizer: HuggingFace tokenizer
        show_labels: Whether to include labels in output

    Returns:
        Dict with decoded 'text' and optional 'labels_text'
    """
    result: dict[str, Any] = {}

    # Decode input_ids
    input_ids = row.get("input_ids", [])
    if input_ids:
        result["text"] = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Decode labels: use input_ids for text, style masked vs loss tokens
    if show_labels and "labels" in row:
        labels = row["labels"]
        masked_positions = [i for i, t in enumerate(labels) if t == -100]
        if masked_positions:
            result["masked_positions_count"] = len(masked_positions)

        # Build styled Text by decoding contiguous segments from input_ids
        styled = Text()
        i = 0
        while i < len(labels):
            is_masked = labels[i] == -100
            j = i
            while j < len(labels) and (labels[j] == -100) == is_masked:
                j += 1
            segment_text = tokenizer.decode(input_ids[i:j], skip_special_tokens=False)
            styled.append(segment_text, style="dim" if is_masked else "bold green")
            i = j
        result["labels_text_styled"] = styled

    # Include other metadata
    for key in ["attention_mask", "position_ids"]:
        if key in row:
            result[key] = row[key]

    return result


def _load_json_folder(path: Path) -> list[tuple[str | None, int, dict[str, Any]]]:
    """Load all JSON files in folder as separate items."""
    items: list[tuple[str | None, int, dict[str, Any]]] = []
    json_files = sorted(path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in directory: {path}")

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            item = json.load(f)
        if not isinstance(item, dict):
            raise ValueError(f"File {json_file.name} is not a JSON object")
        items.append((json_file.stem, 0, item))

    return items


# =============================================================================
# Message Normalization
# =============================================================================


def normalize_messages(
    row: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    input_format: str = "auto",
) -> list[dict[str, Any]]:
    """
    Extract and normalize messages from various formats.

    Supported formats:
    - {"messages": [{"role": ..., "content": ...}]}  (OpenAI/ChatML)
    - {"conversations": [{"from": ..., "value": ...}]}  (ShareGPT)
    - [{"role": ..., "content": ...}]  (raw message list)
    - {"input_messages": ..., "reasoning_turn": ...}  (CanonicalReasoningRecord)

    Args:
        row: Data row containing messages, or raw message list
        input_format: "auto", "chatml", or "sharegpt"

    Returns:
        List of message dicts with "role" and "content" keys
    """
    # Handle raw message list FIRST (before dict conversion)
    if isinstance(row, list):
        return _normalize_message_list(row)

    # Convert Mapping to dict for easier handling
    row = dict(row)  # type: ignore[arg-type]

    if not isinstance(row, dict):
        raise ValueError(f"Expected dict or list, got {type(row).__name__}")

    # Auto-detect format
    if input_format == "auto":
        input_format = _detect_format(row)

    # Apply format-specific normalization
    if input_format == "sharegpt":
        return _normalize_sharegpt(row)

    # Try messages field first
    if "messages" in row:
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        return _normalize_message_list(messages)

    # Try CanonicalReasoningRecord handling
    if "input_messages" in row and "reasoning_turn" in row:
        try:
            from share_core.reasoning_turn import CanonicalReasoningRecord

            canonical = CanonicalReasoningRecord.model_validate(row)
            return canonical.to_messages(include_trace_fields=True)
        except ImportError:
            pass

    # Try other common field names
    for key in ("outbound_messages", "input_messages", "conversations"):
        if key in row:
            messages = row[key]
            if isinstance(messages, str):
                messages = json.loads(messages)
            try:
                return _normalize_message_list(messages)
            except ValueError:
                continue

    raise ValueError(
        f"Could not find messages in row. Expected 'messages', 'conversations', "
        f"or 'input_messages' field. Got keys: {list(row.keys())}"
    )


def _detect_format(row: dict[str, Any]) -> str:
    """Detect the format of the row."""
    if "conversations" in row and "messages" not in row:
        return "sharegpt"
    return "chatml"


def _normalize_message_list(messages: Any) -> list[dict[str, Any]]:
    """Normalize a list of messages to ChatML format."""
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        raise ValueError("Messages must be a list")

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            raise ValueError(f"Message must be a dict, got {type(msg).__name__}")

        # Handle ShareGPT-style {from, value} format
        if "from" in msg and "value" in msg:
            role = msg["from"]
            content = msg["value"]
        elif "role" in msg:
            role = msg["role"]
            content = msg.get("content", "")
        else:
            raise ValueError(
                f"Message missing 'role' or 'from' field: {list(msg.keys())}"
            )

        normalized.append(
            {
                "role": str(role),
                "content": content,
                **{
                    k: v
                    for k, v in msg.items()
                    if k not in {"role", "content", "from", "value"}
                },
            }
        )

    return normalized


def _normalize_sharegpt(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize ShareGPT format to ChatML."""
    messages: list[dict[str, Any]] = []

    # Add system message if present
    system_msg = row.get("system", "")
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    # Convert conversations
    conversations = row.get("conversations", [])
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()

    if not conversations:
        raise ValueError("ShareGPT item has no conversations")

    for conv in conversations:
        if not isinstance(conv, Mapping):
            raise ValueError(f"Conversation must be a dict, got {type(conv).__name__}")

        role = conv.get("from", conv.get("role", ""))
        content = conv.get("value", conv.get("content", ""))
        messages.append({"role": str(role), "content": content})

    return messages


def extract_tools(row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Extract tools from row if present."""
    tools = row.get("tools")
    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    if not isinstance(tools, list):
        return None
    return tools


# =============================================================================
# Sampling Functions
# =============================================================================


def sample_items(
    items: Sequence[tuple[str | None, int, dict[str, Any]]],
    *,
    count: int,
    seed: int,
) -> list[tuple[str | None, int, dict[str, Any]]]:
    """Sample items randomly."""
    if count < 0:
        raise ValueError("--count must be non-negative.")
    if not items or count == 0:
        return []
    if count >= len(items):
        return list(items)

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(items)), count))
    return [items[index] for index in selected_indices]


# =============================================================================
# Display Functions
# =============================================================================


def _format_content(content: Any) -> str:
    """Format content for display."""
    if isinstance(content, str):
        return content.strip() or "<empty>"
    return json.dumps(content, ensure_ascii=False, indent=2)


def _role_style(role: str) -> str:
    """Get color style for role."""
    normalized = role.strip().lower()
    if normalized == "system":
        return "bold cyan"
    if normalized == "user":
        return "bold green"
    if normalized == "assistant":
        return "bold yellow"
    return "bold red"


def _print_metadata(
    console: Console,
    row: Mapping[str, Any],
    *,
    source_name: str | None,
    row_index: int,
) -> None:
    """Print metadata about the row."""
    if source_name is not None:
        console.print(f"Source: {source_name} row={row_index}", soft_wrap=True)
    else:
        console.print(f"Row: {row_index}", soft_wrap=True)

    if "id" in row:
        console.print(f"ID: {row['id']}", soft_wrap=True)
    if "source_name" in row:
        console.print(f"Source: {row['source_name']}", soft_wrap=True)


def _build_message_panel(message: Mapping[str, Any], *, msg_index: int) -> Panel:
    """Build a Rich panel for a single message."""
    role = str(message.get("role", "")).strip() or "<missing>"
    role_header = Text.assemble(
        ("Role: ", "bold white"),
        (role, _role_style(role)),
        ("   ", ""),
        (f"Turn {msg_index}", "dim"),
    )
    body_parts: list[Any] = [role_header, Text(_format_content(message.get("content")))]

    # Show extra fields (like reasoning_content)
    extras = {
        key: value for key, value in message.items() if key not in {"role", "content"}
    }
    if extras:
        body_parts.append(Text("\nExtras", style="dim bold"))
        body_parts.append(
            Text(json.dumps(extras, ensure_ascii=False, indent=2), style="dim")
        )

    return Panel(
        Group(*body_parts),
        border_style=_role_style(role),
        padding=(0, 1),
    )


def _render_messages(
    console: Console,
    messages: Sequence[Mapping[str, Any]],
    *,
    heading: str,
) -> None:
    """Render messages to console."""
    console.print(Text(heading, style="bold magenta"))
    for msg_index, message in enumerate(messages):
        console.print(_build_message_panel(message, msg_index=msg_index))


def _render_tools(console: Console, tools: list[dict[str, Any]]) -> None:
    """Render tools to console."""
    console.print(Text("Tools", style="bold blue"))
    for idx, tool in enumerate(tools):
        tool_name = tool.get("function", {}).get("name", f"Tool {idx}")
        console.print(
            Panel(
                Text(json.dumps(tool, ensure_ascii=False, indent=2)),
                title=tool_name,
                border_style="blue",
                padding=(0, 1),
            )
        )


def _row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract metadata (non-message fields) from row."""
    excluded = {
        "messages",
        "input_messages",
        "outbound_messages",
        "reasoning_turn",
        "conversations",
        "system",
        "tools",
        "with_expected_answer",
        "without_expected_answer",
        # Tokenized fields
        "input_ids",
        "attention_mask",
        "position_ids",
        "labels",
        "images",
        "videos",
        "audios",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def _extract_branch_messages(row: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract branch messages if present."""
    branch_messages: dict[str, list[dict[str, Any]]] = {}
    for key in ("with_expected_answer", "without_expected_answer"):
        value = row.get(key)
        if not isinstance(value, Mapping):
            continue
        try:
            branch_messages[key] = normalize_messages(value)
        except ValueError:
            continue
    return branch_messages


def print_item(
    *,
    console: Console,
    item_number: int,
    total_items: int,
    dataset_index: int,
    source_name: str | None,
    row_index: int,
    row: Mapping[str, Any],
    show_tools: bool = False,
) -> None:
    """Print a single item with all its messages."""
    branch_messages = _extract_branch_messages(row)

    console.rule(
        Text(
            f"Item {item_number}/{total_items} (idx={dataset_index})", style="bold blue"
        )
    )
    _print_metadata(console, row, source_name=source_name, row_index=row_index)

    # Show metadata
    metadata = _row_metadata(row)
    if metadata:
        console.print(
            Panel(
                Text(json.dumps(metadata, ensure_ascii=False, indent=2), style="dim"),
                title="Metadata",
                border_style="dim",
                padding=(0, 1),
            )
        )

    # Show tools if requested
    if show_tools:
        tools = extract_tools(row)
        if tools:
            _render_tools(console, tools)

    # Show messages
    if branch_messages:
        for branch_name, messages in branch_messages.items():
            _render_messages(console, messages, heading=f"Messages ({branch_name})")
    else:
        messages = normalize_messages(row)
        _render_messages(console, messages, heading="Messages")

    console.print()


def print_tokenized_item(
    *,
    console: Console,
    item_number: int,
    total_items: int,
    dataset_index: int,
    source_name: str | None,
    row_index: int,
    row: Mapping[str, Any],
) -> None:
    """Print a single tokenized item with decoded text."""
    console.rule(
        Text(
            f"Item {item_number}/{total_items} (idx={dataset_index})", style="bold blue"
        )
    )
    _print_metadata(console, row, source_name=source_name, row_index=row_index)

    # Show token statistics
    input_ids = row.get("input_ids", [])
    labels = row.get("labels", [])
    attention_mask = row.get("attention_mask", [])

    stats_text = f"Input length: {len(input_ids)}"
    if labels:
        masked_count = sum(1 for t in labels if t == -100)
        stats_text += f" | Labels: {len(labels)} ({masked_count} masked)"
    if attention_mask:
        stats_text += f" | Attention: {sum(attention_mask)}/{len(attention_mask)}"

    console.print(Text(stats_text, style="dim"))

    # Show decoded text
    if "input_text" in row:
        console.print(
            Panel(
                Text(row["input_text"], style="cyan"),
                title="Decoded Input",
                border_style="cyan",
                padding=(0, 1),
            )
        )

    if "labels_text_styled" in row:
        console.print(
            Panel(
                row["labels_text_styled"],
                title="Decoded Labels (loss)",
                border_style="green",
                padding=(0, 1),
            )
        )

    console.print()


def _wait_for_next_sample(
    console: Console, *, item_number: int, total_items: int, total_in_dataset: int, current_index: int
) -> tuple[bool, int | None]:
    """
    Wait for user input before showing next sample.

    Returns:
        (continue_loop, goto_index)
        - continue_loop: True if should continue, False if quit
        - goto_index: None for next item, or specific index to jump to
    """
    if item_number >= total_items:
        return False, None
    response = (
        console.input(
            f"[dim]Current idx={current_index} | Press Enter for next, 'g' to goto (0-{total_in_dataset - 1}), 'q' quit: [/]"
        )
        .strip()
        .lower()
    )
    if response == "q":
        return False, None
    if response == "g":
        try:
            idx_str = console.input(f"[dim]Enter index (0-{total_in_dataset - 1}): [/]")
            idx = int(idx_str.strip())
            if 0 <= idx < total_in_dataset:
                return True, idx
            console.print(f"[red]Invalid index. Must be 0-{total_in_dataset - 1}[/]")
            return True, None
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/]")
            return True, None
    return True, None


# =============================================================================
# Main Entry Point
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    args = parse_args(argv)
    source_path = Path(args.source)

    console = Console()

    # Load data
    items = load_data(source_path)
    if not items:
        raise ValueError(f"No items found in: {source_path}")

    # Check if data is tokenized
    is_tokenized = items and _is_tokenized_row(items[0][2])

    tokenizer = None
    if is_tokenized:
        if not args.tokenizer:
            raise ValueError(
                "Dataset appears to be tokenized (contains 'input_ids'). "
                "Please provide --tokenizer to decode the tokens.\n"
                "Example: viz_chat data/tokenized/ --tokenizer Qwen/Qwen3-8B"
            )
        tokenizer = _get_tokenizer(args.tokenizer)

    # Print summary
    console.print(f"Source: {source_path}", soft_wrap=True)
    if is_tokenized:
        console.print(f"Tokenizer: {args.tokenizer}", soft_wrap=True)
    console.print(f"Total items: {len(items)}", soft_wrap=True)
    console.print(
        "Interactive mode: Enter=next random, 'g'=goto index, 'q'=quit",
        style="dim",
    )
    console.print()

    # Infinite random sampling
    rng = random.Random(args.seed)
    item_number = 1

    while True:
        # Pick a random item
        idx = rng.randint(0, len(items) - 1)
        source_name, row_index, row = items[idx]

        if is_tokenized:
            decoded_row = dict(row)
            decoded_row.update(decode_tokenized_row(row, tokenizer))
            print_tokenized_item(
                console=console,
                item_number=item_number,
                total_items=len(items),
                dataset_index=idx,
                source_name=source_name,
                row_index=row_index,
                row=decoded_row,
            )
        else:
            print_item(
                console=console,
                item_number=item_number,
                total_items=len(items),
                dataset_index=idx,
                source_name=source_name,
                row_index=row_index,
                row=row,
                show_tools=args.show_tools,
            )

        continue_loop, goto_idx = _wait_for_next_sample(
            console,
            item_number=item_number,
            total_items=len(items),
            total_in_dataset=len(items),
            current_index=idx,
        )
        if not continue_loop:
            break

        if goto_idx is not None:
            # Jump to specific index - show it but continue infinite mode after
            source_name, row_index, row = items[goto_idx]
            item_number += 1
            if is_tokenized:
                decoded_row = dict(row)
                decoded_row.update(decode_tokenized_row(row, tokenizer))
                print_tokenized_item(
                    console=console,
                    item_number=item_number,
                    total_items=len(items),
                    dataset_index=goto_idx,
                    source_name=source_name,
                    row_index=row_index,
                    row=decoded_row,
                )
            else:
                print_item(
                    console=console,
                    item_number=item_number,
                    total_items=len(items),
                    dataset_index=goto_idx,
                    source_name=source_name,
                    row_index=row_index,
                    row=row,
                    show_tools=args.show_tools,
                )

        item_number += 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
