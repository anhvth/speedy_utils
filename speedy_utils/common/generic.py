from collections import Counter
from typing import List, TypeVar, Generic, Callable, Any
from threading import Lock
from loguru import logger

T = TypeVar('T')

class TaskDistributor(Generic[T]):
    """Generic class for distributing tasks across multiple workers."""

    def __init__(self, workers: List[T], debug: bool = False):
        """
        Initialize with a list of workers and an optional debug flag.

        :param workers: List of worker instances.
        :param debug: If True, enables debug-level logging.
        """
        self.workers = workers
        self.usage_counter = Counter()  # Tracks the number of tasks each worker is handling
        self.lock = Lock()  # Ensures thread-safe access to shared data

        # Set up logger with loguru
        self.debug = debug
        if self.debug:
            logger.enable(__name__)  # Enable logging for this module
            logger.debug("Logger initialized in debug mode.")
        else:
            logger.disable(__name__)  # Disable logging for this module when debug is False
    def _get_least_busy_worker(self) -> T:
        """Select the worker that has handled the fewest tasks."""
        with self.lock:
            worker = min(self.workers, key=lambda worker: self.usage_counter[worker])

            logger.debug(f"Selected worker: {worker}, usage_counter: {self.usage_counter}")
            return worker

    def _update_worker_usage(self, worker: T, delta: int):
        """Update the usage count of a worker."""
        with self.lock:
            self.usage_counter[worker] += delta
            action = "Incremented" if delta > 0 else "Decremented"
            logger.debug(f"{action} usage count for worker {worker}: {self.usage_counter[worker]}")

    def delegate_task(self, task_func: Callable[[T], Any], *args, **kwargs) -> Any:
        """
        Delegate a task to the least busy worker.

        :param task_func: The task function to execute.
        :param args: Positional arguments for the task function.
        :param kwargs: Keyword arguments for the task function.
        :return: The result of the task function.
        """
        worker = self._get_least_busy_worker()
        # self._update_worker_usage(worker, 1)  # Increment the usage counter
        logger.debug(f"Delegating task {task_func.__name__} to worker {worker} with args {args} and kwargs {kwargs}")
        try:
            result = task_func(worker, *args, **kwargs)
            logger.debug(f"Task {task_func.__name__} completed by worker {worker} with result {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing task {task_func.__name__} with worker {worker}: {e}")
            raise
        # finally:
        #     self._update_worker_usage(worker, -1)  # Decrement the usage counter

    def __getattr__(self, name: str):
        """
        Dynamically delegates the method to the least busy worker.
        Called when an attribute is not found in TaskDistributor.

        :param name: The attribute name to delegate.
        :return: The delegated attribute or method.
        """
        worker = self._get_least_busy_worker()
        if hasattr(worker, name):
            attr = getattr(worker, name)
            # If it's a callable (method), return a wrapper to delegate the task
            if callable(attr):
                def delegated_method(*args, **kwargs):
                    logger.debug(f"Delegating method '{name}' to worker {worker} with args {args} and kwargs {kwargs}")
                    return self.delegate_task(attr, *args, **kwargs)
                return delegated_method
            else:

                logger.debug(f"Accessing attribute '{name}' of worker {worker}")
                return attr  # If it's not callable, return the attribute directly
        error_message = f"'{type(self).__name__}' object has no attribute '{name}'"
        logger.error(error_message)
        raise AttributeError(error_message)

    def __dir__(self):
        """
        Include methods of the workers in TaskDistributor's `dir()`.
        This helps support autocompletion in IDEs.

        :return: Combined list of TaskDistributor and worker attributes.
        """
        worker_methods = dir(self._get_least_busy_worker())
        combined_dir = list(set(super().__dir__() + worker_methods))
        if self.debug:
            logger.debug(f"Combined dir includes: {combined_dir}")
        return combined_dir
