"""Benchmark speedy_utils multi_thread with a simple sleep workload.

Example
-------
uv run python debug/multi_thread.py --rounds 1000 --workers 10 --sleep-s 0.1
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from loguru import logger

from speedy_utils.multi_worker.thread import multi_thread


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level='DEBUG')

    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--sleep-s', type=float, default=0.1)
    args = parser.parse_args()
    # print args
    print(args)

    if args.rounds <= 0:
        raise SystemExit('--rounds must be > 0')
    if args.workers <= 0:
        raise SystemExit('--workers must be > 0')
    if args.sleep_s < 0:
        raise SystemExit('--sleep-s must be >= 0')

    sleep_s = args.sleep_s
    first_item_event = threading.Event()
    first_item_lock = threading.Lock()
    start_t = time.perf_counter()

    logger.debug(
        'multi_thread start rounds={} workers={} sleep_s={}',
        args.rounds,
        args.workers,
        sleep_s,
    )

    def _sleep_fn(_: int) -> None:
        time.sleep(sleep_s)
        if not first_item_event.is_set():
            with first_item_lock:
                if not first_item_event.is_set():
                    first_item_event.set()
                    logger.debug(
                        'time to first item: {:.3f}s',
                        time.perf_counter() - start_t,
                    )

    t0 = start_t
    multi_thread(
        _sleep_fn,
        range(args.rounds),
        workers=args.workers,
        progress=True,
    )
    elapsed_s = time.perf_counter() - t0

    ideal_s = (args.rounds * sleep_s) / args.workers
    print(
        f'rounds={args.rounds} workers={args.workers} sleep_s={sleep_s} '
        f'elapsed_s={elapsed_s:.3f} ideal_s={ideal_s:.3f}'
    )


if __name__ == '__main__':
    main()
