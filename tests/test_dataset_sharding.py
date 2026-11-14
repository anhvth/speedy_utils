"""Tests for multi_process_dataset shard merging."""

import pytest

import speedy_utils.multi_worker.dataset_sharding as sharding


def test_merge_is_refused_when_a_shard_returned_none(monkeypatch):
    monkeypatch.setattr(sharding, '_prepare_dataset', lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(sharding, '_build_shard_jobs', lambda **kwargs: [])
    monkeypatch.setattr(
        sharding, '_run_shard_jobs', lambda jobs, size, workers: ['a', None]
    )
    merges = []
    monkeypatch.setattr(sharding, '_merge_shards', lambda *args: merges.append(args))

    with pytest.raises(RuntimeError, match='produced no output'):
        sharding.multi_process_dataset(
            [], lambda shard, path: path, 'out', num_workers=2
        )

    assert merges == []
