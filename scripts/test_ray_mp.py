from os import getpid
from random import choices
from time import sleep

from speedy_utils import multi_process


def square_value(number: int) -> int:
    """Example worker that squares a number while logging occasionally."""
    if choices([True, False], weights=[0.5, 0.9])[0]:
        print(f"Processing {number} in process {getpid()}")
    sleep(0.1)
    return number * number


def main() -> None:
    """Demonstrate running :func:`square_value` across several worker processes."""
    values = list(range(100))
    worker_count = 4

    multi_process(
        square_value,
        values,
        workers=worker_count,
        backend='mp',
        log_worker='first',
    )


if __name__ == "__main__":
    main()
