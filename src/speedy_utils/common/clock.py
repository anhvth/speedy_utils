import inspect
import os
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
        logger.opt(depth=2).info(
            f"{func.__name__} took {execution_time:0.2f} seconds to execute."
        )
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
        self.min_depth = float("inf")

    def start(self):
        """Start the timer or reset if already started."""
        if self.start_time is not None:
            raise ValueError("Timer has already been started.")
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        # logger.opt(depth=2).info(f"Timer started. {id(self)=}")

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
            logger.opt(depth=2).info(msg)

    def _tick(self):
        """Return the time elapsed since the last checkpoint and update the last checkpoint."""
        # assert self.start_time is not None, f"Timer has not been started. {id(self)=}"
        if not self.start_time:
            logger.opt(depth=2).warning(
                "Timer has not been started. Please call start() before using this method."
            )
            return
        current_time = time.time()
        if self.last_checkpoint is None:
            logger.opt(depth=2).warning(
                "Last checkpoint is not set. Please call start() before using this method."
            )
            return
        elapsed = current_time - self.last_checkpoint
        self.last_checkpoint = current_time
        return elapsed

    def tick(self):
        return self._tick()

    def time_since_last_checkpoint(self):
        """Return the time elapsed since the last checkpoint."""
        if self.start_time is None:
            # raise ValueError("Timer has not been started.")
            logger.opt(depth=2).warning(
                "Timer has not been started. Please call start() before using this method."
            )
            return
        if self.last_checkpoint is None:
            logger.opt(depth=2).warning(
                "Last checkpoint is not set. Please call start() before using this method."
            )
            return
        return time.time() - self.last_checkpoint

    def update_task(self, task_name):
        """Update the elapsed time for the specified task, including file, line, and call depth."""

        # Get the full call stack
        stack = inspect.stack()

        # Get the file and line number of the caller (the previous frame in the stack)
        caller_frame = stack[1]
        file_lineno = f"{os.path.basename(caller_frame.filename)}:{caller_frame.lineno}"

        # Calculate the depth of the current call (i.e., how far it is in the stack)
        call_depth = (
            len(stack) - 1
        )  # Subtract 1 to exclude the current frame from the depth count
        if call_depth < self.min_depth:
            self.min_depth = call_depth

        # Update the task time in the internal task table
        if task_name not in self.task_times:
            self.task_times[task_name] = {
                "time": 0,
                "file_lineno": file_lineno,
                "depth": call_depth,
            }
        self.task_times[task_name]["time"] += self.tick()

    def get_percentage_color(self, percentage):
        """Return ANSI color code based on percentage."""
        if percentage >= 75:
            return "\033[91m"  # Red
        elif percentage >= 50:
            return "\033[93m"  # Yellow
        elif percentage >= 25:
            return "\033[92m"  # Green
        else:
            return "\033[94m"  # Blue

    def print_task_table(self, interval=1, max_depth=None):
        """Print the task time table at regular intervals."""
        current_time = time.time()

        if current_time - self.last_print_time > interval:
            self.print_counter += 1
            total_time = (
                sum(data["time"] for data in self.task_times.values()) or 1
            )  # Avoid division by zero

            # Prepare data for the table
            table_data = []
            for task_name, data in self.task_times.items():
                time_spent = data["time"]
                file_lineno = data["file_lineno"]
                depth = data["depth"] - self.min_depth
                if max_depth is not None and depth > max_depth:
                    continue
                percentage = (time_spent / total_time) * 100

                # Get color code based on percentage
                color_code = self.get_percentage_color(percentage)
                percentage_str = f"{percentage:.2f} %"
                colored_percentage = f"{color_code}{percentage_str}\033[0m"

                table_data.append(
                    [
                        task_name,
                        file_lineno,
                        # depth,
                        f"{time_spent:.2f} s",
                        colored_percentage,
                    ]
                )

            # Add headers and log using tabulate
            table = tabulate(
                table_data,
                headers=["Task", "File:Line", "Time (s)", "Percentage (%)"],
                tablefmt="grid",
            )

            self.last_print_time = current_time
            # total_time_str = f"\nTotal time elapsed: {total_time:.2f} seconds."
            logger.opt(depth=2).info(f"\n{table}")


# Example of how to instantiate the Timer
speedy_timer = Clock(start_now=False)


# Clock, speedy_timer, timef
__all__ = ["Clock", "speedy_timer", "timef"]
