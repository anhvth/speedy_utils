#!/usr/bin/env python
"""
test.py - A simple test suite for the speedy_utils code base.
Run this file to execute tests on various components of the library.
"""

from speedy_utils.all import *
import time

def test_multi_thread_success():
    """Test multi_thread with a function that returns input + 1."""
    def add_one(x):
        time.sleep(0.1)
        return x + 1

    inputs = range(5)
    results = multi_thread(
        func=add_one,
        orig_inputs=inputs,
        workers=2,
        desc="Testing multi_thread success"
    )
    expected = [x + 1 for x in inputs]
    assert results == expected, f"Expected {expected}, got {results}"
    print("multi_thread success test passed.")

def test_multi_thread_error():
    """Test multi_thread with a function that raises an error for a specific input."""
    def error_func(x):
        time.sleep(0.1)
        if x == 3:
            raise ValueError("Test induced error")
        return x

    try:
        # With stop_on_error=True the first error should stop processing
        multi_thread(
            func=error_func,
            orig_inputs=range(5),
            workers=2,
            desc="Testing multi_thread error",
            stop_on_error=True
        )
    except ValueError as e:
        print("Caught expected exception in multi_thread error test:", e)
    else:
        raise AssertionError("Expected exception was not raised in multi_thread error test.")

def test_multi_process_success():
    """Test multi_process with a function that multiplies input by a factor."""
    def multiply(x, factor=3):
        time.sleep(0.1)
        return x * factor

    # Wrap multiply in a lambda to supply 'factor'
    inputs = range(5)
    results = multi_process(
        func=lambda x: multiply(x, factor=3),
        items=inputs,
        workers=2,
        verbose=False
    )
    expected = [x * 3 for x in inputs]
    assert results == expected, f"Expected {expected}, got {results}"
    print("multi_process success test passed.")

def test_memoize():
    """Test memoize decorator with a recursive Fibonacci function."""
    call_count = 0

    @memoize
    def fib(n):
        nonlocal call_count
        call_count += 1
        if n < 2:
            return n
        return fib(n-1) + fib(n-2)

    result = fib(10)
    expected = 55
    assert result == expected, f"Expected {expected}, got {result}"
    # With memoization, the call count should be much lower than without it.
    print(f"Memoize test passed, call_count: {call_count}")

def test_clock_and_timef():
    """Test Clock and timef decorator."""
    @timef
    def dummy():
        time.sleep(0.2)
        return "done"

    start = time.time()
    result = dummy()
    elapsed = time.time() - start
    assert result == "done", "dummy function did not return expected result"
    assert elapsed >= 0.2, "dummy function did not sleep long enough"
    print("Clock and timef test passed.")

def test_task_distributor():
    """Test TaskDistributor with dummy workers."""
    class DummyWorker:
        def __init__(self, name):
            self.name = name
        def process(self, x):
            return f"{self.name} processed {x}"
        def __str__(self):
            return self.name
        __repr__ = __str__

    workers = [DummyWorker("Worker_A"), DummyWorker("Worker_B")]
    distributor = TaskDistributor(workers, debug=True)
    result = distributor.process("test input")
    assert "processed" in result, f"Expected 'processed' in result, got {result}"
    print("TaskDistributor test passed.")

def test_args_parser():
    """Test ArgsParser using ExampleArgs dataclass."""
    from speedy_utils.common.dataclass_parser import ExampleArgs
    args = ExampleArgs()  # Using default values
    expected_model = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    assert args.model_name_or_path == expected_model, f"Expected model_name_or_path to be {expected_model}"
    print("ArgsParser test passed.")

def main():
    print("Starting tests for speedy_utils code base...")
    test_multi_thread_success()
    test_multi_thread_error()
    test_multi_process_success()
    test_memoize()
    test_clock_and_timef()
    test_task_distributor()
    test_args_parser()
    print("All tests passed.")

if __name__ == '__main__':
    main()
