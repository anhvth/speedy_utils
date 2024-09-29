
import time

from loguru import logger
from tabulate import tabulate

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
    """
    A simple clock utility to measure and log time intervals.

    Usage:

    1. Creating and starting the clock:
        clock = Clock(start_now=True)
        # or
        clock = Clock(start_now=False)
        clock.start()

    2. Measure time since the clock started:
        elapsed_time = clock.since_start()

    3. Log the time elapsed since the clock started:
        clock.log()
        # or use a custom logger
        clock.log(custom_logger=my_custom_logger)

    4. Measure time since the last checkpoint:
        time_since_last_check = clock.since_last_check()

    5. Update a named timer in the internal time table:
        clock.update("timer_name")

    6. Print the time table every 'every' seconds:
        clock.print_table(every=1)
    """
    def __init__(self, start_now=True):
        self.start_time = None
        self.time_table = {}
        self.last_check = None
        if start_now:
            self.start()
        self.pbar_counter = 0
        self.last_print = time.time()

    def start(self):
        self.start_time = time.time() if self.start_time is None else self.start_time
        self.last_check = self.start_time

    def since_start(self):
        if self.start_time is None:
            raise ValueError("Clock has not been started.")
        return time.time() - self.start_time

    def log(self, custom_logger=None):
        msg = f"Time elapsed: {self.since_start():.2f} seconds."
        if custom_logger:
            custom_logger(msg)
        else:
            logger.info(msg)

    def since_last_check(self):
        now = time.time()
        elapsed = now - self.last_check
        self.last_check = now
        return elapsed

    def update(self, name):
        if name not in self.time_table:
            self.time_table[name] = 0
        self.time_table[name] += self.since_last_check()

    def print_table(self, every=1):
        now = time.time()
        if now - self.last_print > every:
            self.pbar_counter += 1
            total_time = sum(self.time_table.values())
            
            # Prepare data for the table
            table_data = [
                [name, f"{t:.2f} s", f"{(t / total_time) * 100:.2f} %"] 
                for name, t in self.time_table.items()
            ]
            
            # Add headers and log using tabulate
            table = tabulate(
                table_data, 
                headers=["Task", "Time (s)", "Percentage (%)"],
                tablefmt="grid"  # Grid format for better readability
            )
            
            logger.info(f"\n{table}")
            self.last_print = now
