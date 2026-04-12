"""Debug script for multi_process — no __main__ guard needed.

Usage:
    uv run python scripts/debug_multi_process.py
"""

from time import sleep

from speedy_utils import multi_process

a = 1
def f(x: int, y=3) -> int:
    sleep(0.1)
    return x * 2+a+y


# if __name__ == "__main__":
items = list(range(20))
results = multi_process(f, items, num_procs=4, desc="debug sleep", y=3)
print(f"results: {results}")
