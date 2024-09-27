
import time
from loguru import logger

__all__ = ["Clock", "timef"]


def timef(func):
    "Decorator to print the execution time of a function"
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:0.2f} seconds to execute.")
        return result

    return wrapper


class Clock:
    def __init__(self, start_now=True):
        "Initialize the clock, optionally starting it."
        self.start_time = None
        self.time_table = {}
        self.last_check = None
        if start_now:
            self.start()
        self.pbar_counter = 0
        self.last_print = time.time()

    def start(self):
        "Start the clock."
        self.start_time = time.time() if self.start_time is None else self.start_time
        self.last_check = self.start_time

    def since_start(self):
        "Return the time elapsed since the clock started."
        if self.start_time is None:
            raise ValueError("Clock has not been started.")
        return time.time() - self.start_time

    def log(self, custom_logger=None):
        "Log the time elapsed since the clock started."
        msg = f"Time elapsed: {self.since_start():.2f} seconds."
        if custom_logger:
            custom_logger(msg)
        else:
            logger.info(msg)

    def since_last_check(self):
        "Return the time elapsed since the last check."
        now = time.time()
        elapsed = now - self.last_check
        self.last_check = now
        return elapsed

    def update(self, name):
        "Update the time table with the elapsed time since the last check."
        if not name in self.time_table:
            self.time_table[name] = 0
        self.time_table[name] += self.since_last_check()

    def print_table(self, every=1):
        "Print the time table if the specified interval has passed."
        now = time.time()
        if now - self.last_print > every:
            self.pbar_counter += 1
            total_time = sum(self.time_table.values())
            desc = "Time table: "
            for name, t in self.time_table.items():
                percentage = (t / total_time) * 100
                desc += "{}: avg_time: {:.2f} s ({:.2f}%), total: {} s".format(
                    name, t, percentage, total_time
                )
            logger.info(desc)
            self.last_print = now
