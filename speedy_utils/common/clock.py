import inspect
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
    A simple timer utility to measure and log time intervals.

    Usage:

    1. Creating and starting the timer:
        timer = Timer(start_now=True)
        # or
        timer = Timer(start_now=False)
        timer.start()

    2. Measure time since the timer started:
        elapsed_time = timer.elapsed_time()

    3. Log the time elapsed since the timer started:
        timer.log_elapsed_time()
        # or use a custom logger
        timer.log_elapsed_time(custom_logger=my_custom_logger)

    4. Measure time since the last checkpoint:
        time_since_last_checkpoint = timer.time_since_last_checkpoint()

    5. Update a named task in the internal task time table:
        timer.update_task("task_name")

    6. Print the task time table every 'interval' seconds:
        timer.print_task_table(interval=1)
    """

    def __init__(self, start_now=True):
        """Initialize the timer and optionally start it immediately."""
        self.start_time = None
        self.task_times = {}
        self.last_checkpoint = None
        if start_now:
            self.start()
        self.print_counter = 0
        self.last_print_time = time.time()

    def start(self):
        """Start the timer or reset if already started."""
        if self.start_time is None:
            self.start_time = time.time()
            self.last_checkpoint = self.start_time
            logger.info(f"Timer started. {id(self)=}")

    def elapsed_time(self):
        """Return the time elapsed since the timer started."""
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        return time.time() - self.start_time

    def log_elapsed_time(self, custom_logger=None):
        """Log the time elapsed since the timer started."""
        msg = f"Time elapsed: {self.elapsed_time():.2f} seconds."
        if custom_logger:
            custom_logger(msg)
        else:
            logger.info(msg)

    def time_since_last_checkpoint(self):
        """Return the time elapsed since the last checkpoint."""
        assert self.start_time is not None, f"Timer has not been started. {id(self)=}"
        current_time = time.time()
        elapsed = current_time - self.last_checkpoint
        self.last_checkpoint = current_time
        return elapsed

    def update_task(self, task_name):
        """Update the elapsed time for the specified task."""
        # get the file:line_no that call this
        file_lineno = f"{inspect.stack()[2].filename.split('/')[-1]}:{inspect.stack()[2].lineno}"

        task_name = f"{task_name} ({file_lineno})"
        if task_name not in self.task_times:
            self.task_times[task_name] = 0
        self.task_times[task_name] += self.time_since_last_checkpoint()

    def print_task_table(self, interval=1):
        """Print the task time table at regular intervals."""
        current_time = time.time()
        if current_time - self.last_print_time > interval:
            self.print_counter += 1
            total_time = sum(self.task_times.values())

            # Prepare data for the table
            table_data = [
                [task_name, f"{time_spent:.2f} s", f"{(time_spent / total_time) * 100:.2f} %"]
                for task_name, time_spent in self.task_times.items()
            ]

            # Add headers and log using tabulate
            table = tabulate(table_data, headers=["Task", "Time (s)", "Percentage (%)"], tablefmt="grid")

            self.last_print_time = current_time
            logger.opt(depth=1).info(f"\n{table}")


# Example of how to instantiate the Timer
speedy_timer = Clock(start_now=False)
