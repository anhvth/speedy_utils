# Multi-thread Error Tracing Improvements

## Summary

Significantly improved error tracing in `multi_thread` to focus on user code rather than infrastructure frames, making debugging much faster and easier.

## Problem

Previously, when errors occurred in functions executed by `multi_thread`, the traceback was cluttered with infrastructure frames:

- `concurrent.futures` internals
- `threading.py` frames
- `multi_worker/thread.py` infrastructure code

This made it difficult to quickly identify the actual problem in user code.

### Example of OLD behavior:

```
TypeError                                 Traceback (most recent call last)
Cell In[810], line 35
     33     choices = multi_thread(fns, range(n))
     34     return choices
---> 35 choices = translate()

File ~/projects/speedy_utils/src/speedy_utils/multi_worker/thread.py:474, in multi_thread(...)
    472 idx, logical_size = _future_meta(fut)
    473 try:
--> 474     result = fut.result()
    475 except Exception as exc:
    476     if stop_on_error:

File ~/.local/share/uv/python/.../concurrent/futures/_base.py:449, in Future.result(...)
    447     raise CancelledError()
    448 elif self._state == FINISHED:
--> 449     return self.__get_result()

... (many more infrastructure frames) ...

TypeError: 'list' object is not callable
```

## Solution

### 1. Added `UserFunctionError` Exception Class

A custom exception wrapper that:

- Captures the original exception
- Stores the function name and input that caused the error
- Filters traceback to include only user code frames
- Provides clear, focused error messages

### 2. Enhanced `_worker` Function

- Added validation to detect common mistakes (e.g., passing a list instead of a function)
- Filters tracebacks to remove infrastructure frames
- Wraps user function errors in `UserFunctionError` with clean context
- Provides helpful hints for common mistakes

### 3. Improved Error Reporting in `multi_thread`

- Logs clear error messages showing function name and input
- Displays only user code in tracebacks
- Re-raises exceptions with cleaned messages
- Maintains proper exception chaining while hiding infrastructure noise

## Benefits

### Clear Error Messages

```
Error in function "process_item" with input: 0

User code traceback:
  File "your_script.py", line 20, in process_item
    return my_list(x)
           ^^^^^^^^^^
TypeError: 'list' object is not callable
```

### Helpful Hints

```
TypeError:
multi_thread: func parameter must be callable, got list: [...]
Hint: Did you accidentally pass a list instead of a function?
```

### Nested Function Support

Shows complete call chain through user code:

```
Error in function "process_data" with input: 0

User code traceback:
  File "your_script.py", line 44, in process_data
    return validate_and_calc(val)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "your_script.py", line 42, in validate_and_calc
    return 100 / x
           ~~~~^~~
ZeroDivisionError: division by zero
```

## Key Improvements

✅ **Errors show function name and problematic input**
✅ **Tracebacks filtered to show only user code**
✅ **No concurrent.futures/threading clutter**
✅ **Helpful hints for common mistakes**
✅ **Clear, actionable error messages**
✅ **Maintains backward compatibility - all existing tests pass**

## Testing

Run the comprehensive demo to see all improvements:

```bash
python tests/test_multithread_error_trace.py
```

This demonstrates:

1. Simple function errors
2. Nested function call traces
3. Common parameter type mistakes
4. Various exception types (TypeError, ValueError, AttributeError, etc.)

## Code Changes

Main files modified:

- `src/speedy_utils/multi_worker/thread.py`:
  - Added `UserFunctionError` exception class
  - Enhanced `_worker` function with validation and error filtering
  - Improved error handling in `multi_thread` main loop
  - Added imports for `sys` and `traceback`

All changes maintain backward compatibility - existing code continues to work unchanged.
