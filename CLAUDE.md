# Claude Code Instructions for speedy_utils

## Performance Requirement: Import Time < 0.4s

**CRITICAL:** All modules in this project MUST import in under 0.4 seconds. This is enforced by a git pre-commit hook that will block commits if import time exceeds 0.4s.

### Why This Matters

- Users should be able to `import speedy_utils` instantly
- Slow imports hurt developer experience and productivity
- Heavy dependencies like torch, ray, matplotlib should NOT load until actually used

### Lazy Import Strategy

**Rule:** Import heavy modules inside the function that needs them, NOT at module level.

```python
# BAD - Imports at module level (slow!)
import torch
import ray
import matplotlib.pyplot as plt

def some_function():
    return torch.tensor([1, 2, 3])


# GOOD - Lazy import inside function (fast!)
def some_function():
    import torch  # Only imports when function is called
    return torch.tensor([1, 2, 3])
```

### Heavy Modules to Lazy-Load

| Module | Typical Import Time | Strategy |
|--------|---------------------|----------|
| `torch` | ~5s cumulative | Lazy import in GPU functions |
| `ray` | ~10s cumulative | Lazy import in distributed functions |
| `matplotlib` | ~3.6s cumulative | Lazy import in plotting functions |
| `pandas` | ~1.3s cumulative | Lazy import in data functions |
| `IPython` | ~1s cumulative | Lazy import in notebook functions |

### Using the _LazyModule Pattern

For commonly used heavy modules, use the `_LazyModule` class from `__imports.py`:

```python
from speedy_utils.__imports import pd, ray, plt

# These are lazy - only import when first accessed
def process_data():
    df = pd.DataFrame()  # pandas imports here
    return df
```

### Testing Import Time

To check import time manually:

```bash
# Check specific module
python -c "import time; start=time.perf_counter(); import speedy_utils; print(f'{time.perf_counter()-start:.3f}s')"

# Run detailed analysis
python scripts/debug_import_time.py speedy_utils --min-sec 0.05 --no-stdlib
```

### When Adding New Dependencies

1. Check import time of the new dependency
2. If > 0.1s, use lazy import strategy
3. Test that `import speedy_utils` still completes in < 0.4s
4. The pre-commit hook will enforce this automatically

### Module-Specific Targets

- `speedy_utils`: < 0.4s
- `llm_utils`: < 0.4s
- `vision_utils`: < 0.4s
