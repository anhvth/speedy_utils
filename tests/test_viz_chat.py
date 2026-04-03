"""Tests for viz_chat module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from datasets_utils.viz_chat import (
    _detect_format,
    _load_json,
    _load_json_folder,
    _load_jsonl,
    extract_tools,
    load_data,
    normalize_messages,
    sample_items,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_chatml_messages():
    """Sample ChatML format messages."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]


@pytest.fixture
def sample_sharegpt_item():
    """Sample ShareGPT format item."""
    return {
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "2+2 equals 4."},
        ],
        "system": "You are a math helper.",
    }


@pytest.fixture
def sample_messages_item(sample_chatml_messages):
    """Sample messages format item."""
    return {"messages": sample_chatml_messages}


@pytest.fixture
def sample_tools():
    """Sample tools in OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                },
            },
        }
    ]


# =============================================================================
# Test Data Loading
# =============================================================================


class TestLoadJsonl:
    """Tests for JSONL loading."""

    def test_load_single_item(self, sample_messages_item):
        """Test loading a JSONL file with a single item."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_messages_item))
            f.flush()
            try:
                items = _load_jsonl(Path(f.name))
                assert len(items) == 1
                assert items[0][2]["messages"] == sample_messages_item["messages"]
            finally:
                Path(f.name).unlink()

    def test_load_multiple_items(self):
        """Test loading a JSONL file with multiple items."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(3):
                item = {"messages": [{"role": "user", "content": f"Message {i}"}]}
                f.write(json.dumps(item) + "\n")
            f.flush()
            try:
                items = _load_jsonl(Path(f.name))
                assert len(items) == 3
            finally:
                Path(f.name).unlink()

    def test_load_with_empty_lines(self, sample_messages_item):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_messages_item) + "\n\n\n")
            f.write(json.dumps(sample_messages_item) + "\n")
            f.flush()
            try:
                items = _load_jsonl(Path(f.name))
                assert len(items) == 2
            finally:
                Path(f.name).unlink()

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Invalid JSON"):
                    _load_jsonl(Path(f.name))
            finally:
                Path(f.name).unlink()


class TestLoadJson:
    """Tests for JSON file loading."""

    def test_load_single_object(self, sample_messages_item):
        """Test loading a JSON file with a single object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_messages_item, f)
            f.flush()
            try:
                items = _load_json(Path(f.name))
                assert len(items) == 1
                assert items[0][2] == sample_messages_item
            finally:
                Path(f.name).unlink()

    def test_load_array(self):
        """Test loading a JSON file with an array."""
        data = [
            {"messages": [{"role": "user", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "Bye"}]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            try:
                items = _load_json(Path(f.name))
                assert len(items) == 2
            finally:
                Path(f.name).unlink()

    def test_invalid_type_raises_error(self):
        """Test that non-object/array raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump("just a string", f)
            f.flush()
            try:
                with pytest.raises(ValueError, match="Expected JSON object or array"):
                    _load_json(Path(f.name))
            finally:
                Path(f.name).unlink()


class TestLoadJsonFolder:
    """Tests for JSON folder loading."""

    def test_load_folder(self):
        """Test loading from a folder of JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                item = {"messages": [{"role": "user", "content": f"Item {i}"}]}
                with open(Path(tmpdir) / f"file_{i}.json", "w") as f:
                    json.dump(item, f)

            items = _load_json_folder(Path(tmpdir))
            assert len(items) == 3
            # Check that source names are file stems
            source_names = {item[0] for item in items}
            assert source_names == {"file_0", "file_1", "file_2"}

    def test_empty_folder_raises_error(self):
        """Test that empty folder raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No JSON files found"):
                _load_json_folder(Path(tmpdir))


class TestLoadData:
    """Tests for the main load_data function."""

    def test_load_jsonl_file(self, sample_messages_item):
        """Test load_data with JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_messages_item))
            f.flush()
            try:
                items = load_data(f.name)
                assert len(items) == 1
            finally:
                Path(f.name).unlink()

    def test_load_json_file(self, sample_messages_item):
        """Test load_data with JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_messages_item, f)
            f.flush()
            try:
                items = load_data(f.name)
                assert len(items) == 1
            finally:
                Path(f.name).unlink()

    def test_load_folder(self):
        """Test load_data with folder of JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            item = {"messages": [{"role": "user", "content": "Hi"}]}
            with open(Path(tmpdir) / "chat.json", "w") as f:
                json.dump(item, f)

            items = load_data(tmpdir)
            assert len(items) == 1

    def test_unsupported_extension_raises_error(self):
        """Test that unsupported file extension raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Unsupported file type"):
                    load_data(f.name)
            finally:
                Path(f.name).unlink()

    def test_nonexistent_path_raises_error(self):
        """Test that nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path")


# =============================================================================
# Test Message Normalization
# =============================================================================


class TestNormalizeMessages:
    """Tests for message normalization."""

    def test_normalize_chatml(self, sample_messages_item, sample_chatml_messages):
        """Test normalizing ChatML format."""
        result = normalize_messages(sample_messages_item)
        assert result == sample_chatml_messages

    def test_normalize_sharegpt(self, sample_sharegpt_item):
        """Test normalizing ShareGPT format."""
        result = normalize_messages(sample_sharegpt_item, input_format="sharegpt")
        assert len(result) == 3  # system + 2 conversations
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "human"
        assert result[2]["role"] == "gpt"

    def test_auto_detect_sharegpt(self, sample_sharegpt_item):
        """Test auto-detection of ShareGPT format."""
        result = _detect_format(sample_sharegpt_item)
        assert result == "sharegpt"

    def test_auto_detect_chatml(self, sample_messages_item):
        """Test auto-detection of ChatML format."""
        result = _detect_format(sample_messages_item)
        assert result == "chatml"

    def test_invalid_format_raises_error(self):
        """Test that invalid data raises ValueError."""
        with pytest.raises(ValueError, match="Could not find messages"):
            normalize_messages({"foo": "bar"})

    def test_raw_message_list(self, sample_chatml_messages):
        """Test normalizing raw message list."""
        result = normalize_messages(sample_chatml_messages)
        assert result == sample_chatml_messages

    def test_messages_as_json_string(self):
        """Test normalizing when messages is a JSON string."""
        item = {
            "messages": json.dumps([
                {"role": "user", "content": "Hi"},
            ])
        }
        result = normalize_messages(item)
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestExtractTools:
    """Tests for tool extraction."""

    def test_extract_tools(self, sample_tools):
        """Test extracting tools from item."""
        item = {"messages": [], "tools": sample_tools}
        result = extract_tools(item)
        assert result == sample_tools

    def test_extract_tools_from_string(self, sample_tools):
        """Test extracting tools from JSON string."""
        item = {"messages": [], "tools": json.dumps(sample_tools)}
        result = extract_tools(item)
        assert result == sample_tools

    def test_no_tools_returns_none(self):
        """Test that missing tools returns None."""
        item = {"messages": []}
        result = extract_tools(item)
        assert result is None

    def test_invalid_tools_returns_none(self):
        """Test that invalid tools returns None."""
        item = {"messages": [], "tools": "not a list"}
        result = extract_tools(item)
        assert result is None


# =============================================================================
# Test Sampling
# =============================================================================


class TestSampleItems:
    """Tests for item sampling."""

    def test_sample_subset(self):
        """Test sampling a subset of items."""
        items = [(None, i, {"id": i}) for i in range(10)]
        result = sample_items(items, count=3, seed=42)
        assert len(result) == 3

    def test_sample_all(self):
        """Test sampling all items when count >= len."""
        items = [(None, i, {"id": i}) for i in range(5)]
        result = sample_items(items, count=10, seed=0)
        assert len(result) == 5

    def test_sample_zero(self):
        """Test sampling zero items."""
        items = [(None, i, {"id": i}) for i in range(5)]
        result = sample_items(items, count=0, seed=0)
        assert len(result) == 0

    def test_sample_empty(self):
        """Test sampling from empty list."""
        result = sample_items([], count=5, seed=0)
        assert len(result) == 0

    def test_negative_count_raises_error(self):
        """Test that negative count raises ValueError."""
        items = [(None, 0, {"id": 0})]
        with pytest.raises(ValueError, match="must be non-negative"):
            sample_items(items, count=-1, seed=0)

    def test_reproducible_with_seed(self):
        """Test that same seed produces same results."""
        items = [(None, i, {"id": i}) for i in range(20)]
        result1 = sample_items(items, count=5, seed=123)
        result2 = sample_items(items, count=5, seed=123)
        assert result1 == result2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow_jsonl(self):
        """Test full workflow with JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                item = {
                    "messages": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Answer {i}"},
                    ]
                }
                f.write(json.dumps(item) + "\n")
            f.flush()

            try:
                # Load data
                items = load_data(f.name)
                assert len(items) == 5

                # Sample items
                sampled = sample_items(items, count=2, seed=42)
                assert len(sampled) == 2

                # Normalize messages
                for source_name, row_index, row in sampled:
                    messages = normalize_messages(row)
                    assert len(messages) == 2
                    assert messages[0]["role"] == "user"
                    assert messages[1]["role"] == "assistant"
            finally:
                Path(f.name).unlink()

    def test_full_workflow_json_folder(self):
        """Test full workflow with folder of JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                item = {
                    "conversations": [
                        {"from": "human", "value": f"Q{i}"},
                        {"from": "gpt", "value": f"A{i}"},
                    ]
                }
                with open(Path(tmpdir) / f"chat_{i}.json", "w") as f:
                    json.dump(item, f)

            # Load data
            items = load_data(tmpdir)
            assert len(items) == 3

            # Sample and normalize
            sampled = sample_items(items, count=2, seed=0)
            for source_name, row_index, row in sampled:
                messages = normalize_messages(row, input_format="sharegpt")
                assert len(messages) == 2
