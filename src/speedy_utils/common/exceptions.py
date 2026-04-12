"""
Library-friendly exception hierarchy for speedy_utils.

These exceptions replace ``sys.exit`` / ``SystemExit`` in core library
paths so that downstream callers can catch, retry, or compose operations
without the process being terminated.
"""


class SpeedyExecutionError(Exception):
    """Base exception for speedy_utils runtime failures.

    All library-code failures that previously terminated the process
    (via ``sys.exit``) should raise a subclass of this instead.
    """

    def __init__(self, message: str = "", *, backend: str = "") -> None:
        self.backend = backend
        super().__init__(message)


class SpeedyWorkerError(SpeedyExecutionError):
    """Raised when a worker function fails inside ``multi_process`` or
    ``multi_thread``.

    Preserves the original exception, rich traceback text, and the
    backend that produced the failure (e.g. ``"spawn"``, ``"thread"``).
    """

    def __init__(
        self,
        message: str = "",
        *,
        original_exception: Exception | None = None,
        backend: str = "",
        func_name: str = "",
    ) -> None:
        self.original_exception = original_exception
        self.func_name = func_name
        super().__init__(message, backend=backend)

    def __str__(self) -> str:
        parts: list[str] = []
        if self.backend:
            parts.append(f"[backend={self.backend}]")
        if self.func_name:
            parts.append(f"[func={self.func_name}]")
        base = super().__str__()
        if parts:
            return f"{' '.join(parts)} {base}"
        return base


class SpeedySerializationError(SpeedyExecutionError):
    """Raised when a payload (input, output, or result) cannot be
    serialized for inter-process communication.

    Typically wraps a ``pickle.PicklingError`` with additional context
    about *what* failed to serialize.
    """

    def __init__(
        self,
        message: str = "",
        *,
        original_exception: Exception | None = None,
    ) -> None:
        self.original_exception = original_exception
        super().__init__(message)
