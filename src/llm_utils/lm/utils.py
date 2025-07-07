import fcntl
import os
import tempfile
import time
from typing import List, Dict
import numpy as np
from loguru import logger




def _atomic_save(array: np.ndarray, filename: str):
    tmp_dir = os.path.dirname(filename) or "."
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp:
        np.save(tmp, array)
        temp_name = tmp.name
    os.replace(temp_name, filename)


def _update_port_use(port: int, increment: int) -> None:
    file_counter: str = f"/tmp/port_use_counter_{port}.npy"
    file_counter_lock: str = f"/tmp/port_use_counter_{port}.lock"
    with open(file_counter_lock, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if os.path.exists(file_counter):
                try:
                    counter = np.load(file_counter)
                except Exception as e:
                    logger.warning(f"Corrupted usage file {file_counter}: {e}")
                    counter = np.array([0])
            else:
                counter: np.ndarray = np.array([0], dtype=np.int64)
            counter[0] += increment
            _atomic_save(counter, file_counter)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _pick_least_used_port(ports: List[int]) -> int:
    global_lock_file = "/tmp/ports.lock"
    with open(global_lock_file, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            port_use: Dict[int, int] = {}
            for port in ports:
                file_counter = f"/tmp/port_use_counter_{port}.npy"
                if os.path.exists(file_counter):
                    try:
                        counter = np.load(file_counter)
                    except Exception as e:
                        logger.warning(f"Corrupted usage file {file_counter}: {e}")
                        counter = np.array([0])
                else:
                    counter = np.array([0])
                port_use[port] = counter[0]
            if not port_use:
                if ports:
                    raise ValueError("Port usage data is empty, cannot pick a port.")
                else:
                    raise ValueError("No ports provided to pick from.")
            lsp = min(port_use, key=lambda k: port_use[k])
            _update_port_use(lsp, 1)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return lsp


def retry_on_exception(max_retries=10, exceptions=(Exception,), sleep_time=3):
    def decorator(func):
        from functools import wraps

        def wrapper(self, *args, **kwargs):
            retry_count = kwargs.get("retry_count", 0)
            last_exception = None
            while retry_count <= max_retries:
                try:
                    return func(self, *args, **kwargs)
                except exceptions as e:
                    import litellm # type: ignore

                    if isinstance(
                        e, (litellm.exceptions.APIError, litellm.exceptions.Timeout)
                    ):
                        base_url_info = kwargs.get(
                            "base_url", getattr(self, "base_url", None)
                        )
                        logger.warning(
                            f"[{base_url_info=}] {type(e).__name__}: {str(e)[:100]}, will sleep for {sleep_time}s and retry"
                        )
                        time.sleep(sleep_time)
                        retry_count += 1
                        kwargs["retry_count"] = retry_count
                        last_exception = e
                        continue
                    elif hasattr(
                        litellm.exceptions, "ContextWindowExceededError"
                    ) and isinstance(e, litellm.exceptions.ContextWindowExceededError):
                        logger.error(f"Context window exceeded: {e}")
                        raise
                    else:
                        logger.error(f"Generic error during LLM call: {e}")
                        import traceback

                        traceback.print_exc()
                        raise
            logger.error(f"Retry limit exceeded, error: {last_exception}")
            if last_exception:
                raise last_exception
            raise ValueError("Retry limit exceeded with no specific error.")

        return wraps(func)(wrapper)

    return decorator


def forward_only(func):
    from functools import wraps

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        kwargs["retry_count"] = 0
        return func(self, *args, **kwargs)

    return wrapper
