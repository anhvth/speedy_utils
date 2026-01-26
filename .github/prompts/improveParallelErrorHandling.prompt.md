---
name: improveParallelErrorHandling
description: Enhance error tracebacks in parallel execution with rich formatting and context
argument-hint: the parallel execution function and backend type
---

Improve error handling for the specified parallel execution function to provide clean, user-focused tracebacks similar to direct function calls.

## Requirements

1. **Filter Internal Frames**: Remove framework/library internal frames from tracebacks, showing only user code
2. **Add Context Lines**: Display 3 lines before and after each error location with line numbers
3. **Include Caller Frame**: Show where the parallel execution function was called, not just where the error occurred
4. **Rich Formatting**: Use rich library's Panel/formatting for clean, readable output
5. **Suppress Noise**: Set environment variables or flags to suppress verbose framework error logs

## Implementation Steps

1. **Capture Caller Context**: Use `inspect.currentframe().f_back` to capture where the parallel function was called (filename, line number, function name)

2. **Wrap Error Handling**: Catch framework-specific exceptions (e.g., `RayTaskError`, thread exceptions) in the execution loop

3. **Parse/Extract Original Exception**: Get the underlying user exception from the framework wrapper
   - Extract exception type, message, and traceback information
   - Parse from string representation if traceback objects aren't preserved

4. **Filter Frames**: Skip frames matching internal paths:
   - Framework internals (e.g., `ray/_private`, `concurrent/futures`)
   - Library worker implementations (e.g., `speedy_utils/multi_worker`)
   - Site-packages for the framework

5. **Format with Context**:
   - For each user frame, show: `filepath:lineno in function_name`
   - Use `linecache.getline()` to retrieve surrounding lines
   - Highlight the error line with `❱` marker
   - Number all lines (e.g., `   4 │ code here` or `   5 ❱ error here`)

6. **Display Caller Frame First**: Show where the parallel function was invoked before showing the actual error location

7. **Clean Exit**: Flush output streams before exiting to ensure traceback displays

## Example Output Format

```
╭─────────────── Traceback (most recent call last) ───────────────╮
│ /path/to/user/script.py:42 in main                              │
│                                                                  │
│   40 │ data = load_data()                                        │
│   41 │ # Process in parallel                                     │
│   42 ❱ results = multi_process(process_item, data, workers=8)   │
│   43 │                                                           │
│                                                                  │
│ /path/to/user/module.py:15 in process_item                      │
│                                                                  │
│   12 │ def process_item(item):                                   │
│   13 │     value = item['key']                                   │
│   14 │     denominator = value - 100                             │
│   15 ❱     return 1 / denominator                                │
│   16 │                                                           │
╰──────────────────────────────────────────────────────────────────╯
ZeroDivisionError: division by zero
```

Apply these improvements to the specified parallel execution function, ensuring error messages are as clear as direct function calls while maintaining all performance benefits of parallel execution.
