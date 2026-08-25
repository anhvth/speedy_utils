"""Hermetic contract tests for the ordered Ray task mapper."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from speedy_utils.multi_worker.multi_process_ray import multi_process_ray


def square(value: int, *, offset: int = 0) -> int:
    return value * value + offset


def fail_on_two(value: int) -> int:
    if value == 2:
        raise ValueError("two is invalid")
    return value


@dataclass(frozen=True)
class _Ref:
    value: object
    error: Exception | None = None


class _FakeRay:
    def __init__(self) -> None:
        self.initialized = False
        self.init_addresses: list[str] = []
        self.submitted: list[_Ref] = []
        self.cancelled: list[_Ref] = []
        self.resource_requests: list[dict[str, object]] = []
        self.wait_queue_sizes: list[int] = []

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, *, address: str) -> None:
        self.initialized = True
        self.init_addresses.append(address)

    def cluster_resources(self) -> dict[str, float]:
        return {"CPU": 64.0, "GPU": 2.0}

    def remote(self, **resources: object):
        self.resource_requests.append(resources)

        def decorate(function):
            def submit(*args: object) -> _Ref:
                try:
                    ref = _Ref(function(*args))
                except Exception as error:  # emulate RayTaskError delivery at get().
                    ref = _Ref(None, error)
                self.submitted.append(ref)
                return ref

            return SimpleNamespace(remote=submit)

        return decorate

    def wait(self, refs: list[_Ref], *, num_returns: int) -> tuple[list[_Ref], list[_Ref]]:
        assert num_returns == 1
        self.wait_queue_sizes.append(len(refs))
        return refs[:1], refs[1:]

    @staticmethod
    def get(ref: _Ref) -> object:
        if ref.error:
            raise ref.error
        return ref.value

    def cancel(self, ref: _Ref, *, force: bool) -> None:
        assert force is True
        self.cancelled.append(ref)


@pytest.fixture
def fake_ray(monkeypatch: pytest.MonkeyPatch) -> _FakeRay:
    fake = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    return fake


def test_orders_results_and_uses_importable_worker(fake_ray: _FakeRay) -> None:
    assert multi_process_ray(square, [3, 1, 2], offset=4, progress=False) == [13, 5, 8]
    assert fake_ray.init_addresses == ["auto"]
    assert fake_ray.resource_requests == [
        {"num_cpus": 1, "num_gpus": 0, "runtime_env": None}
    ]


def test_gpu_default_queue_uses_gpu_capacity(fake_ray: _FakeRay) -> None:
    assert multi_process_ray(square, range(10), num_gpus=1, progress=False) == [
        value * value for value in range(10)
    ]
    # Two GPUs x two queued tasks per GPU, not 64 CPUs x four.
    assert fake_ray.wait_queue_sizes[0] == 4


def test_non_importable_function_fails_before_connecting(fake_ray: _FakeRay) -> None:
    with pytest.raises(TypeError, match="top-level function"):
        multi_process_ray(lambda value: value, [1], progress=False)
    assert fake_ray.init_addresses == []


def test_ignore_preserves_failed_result_slot(fake_ray: _FakeRay) -> None:
    assert multi_process_ray(
        fail_on_two, [1, 2, 3], progress=False, error_handler="ignore"
    ) == [1, None, 3]


def test_invalid_progress_increment_cancels_pending(fake_ray: _FakeRay) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        multi_process_ray(
            square,
            [1, 2, 3],
            max_in_flight=3,
            progress=False,
            progress_increment=lambda _item, _result: -1,
        )
    assert len(fake_ray.cancelled) == 2
