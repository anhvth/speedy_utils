import functools
import time
import traceback
from collections.abc import Callable
from typing import Any

from loguru import logger


def retry_runtime(
    sleep_seconds: int = 5,
    max_retry: int = 12,
    exceptions: type[Exception] | tuple[type[Exception], ...] = (RuntimeError,),
) -> Callable:
    """Decorator that retries the function with exponential backoff on specified runtime exceptions.

    Args:
        sleep_seconds (int): Initial sleep time between retries in seconds
        max_retry (int): Maximum number of retry attempts
        exceptions (Union[Type[Exception], Tuple[Type[Exception], ...]]): Exception types to retry on

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_retry + 1):
                try:
                    return func(*args, **kwargs)

                except (SyntaxError, NameError, ImportError, TypeError) as e:
                    # Don't retry on syntax/compilation errors
                    logger.opt(depth=1).error(
                        f"Critical error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                    )
                    raise

                except exceptions as e:
                    if attempt == max_retry:
                        logger.opt(depth=1).error(
                            f"Function {func.__name__} failed after {max_retry} retries: {str(e)}"
                        )
                        raise

                    backoff_time = sleep_seconds * (
                        2 ** (attempt - 1)
                    )  # Exponential backoff
                    logger.opt(depth=1).warning(
                        f"Attempt {attempt}/{max_retry} failed: {str(e)[:100]}. "
                        f"Retrying in {backoff_time} seconds."
                    )
                    time.sleep(backoff_time)

            return None  # This line should never be reached

        return wrapper

    return decorator


__all__ = ["retry_runtime"]
