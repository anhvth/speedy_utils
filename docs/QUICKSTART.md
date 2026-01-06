# Quick Reference: Zero-Copy Sharing

## TL;DR

Share large objects across workers without copying:

```python
from speedy_utils import multi_process
import numpy as np

model = np.random.randn(1000, 1000)  # 8 MB

results = multi_process(
    my_func,
    items,
    backend='ray',
    shared_kwargs=['model'],  # ← Zero-copy magic!
    model=model
)
```

**Result**: 15x faster, 75% less memory 🚀

## Quick Examples

### Basic Usage
```python
def process(id, model=None):
    return model[id % 100, 0]

multi_process(
    process, 
    range(100),
    backend='ray',
    shared_kwargs=['model'],
    model=large_model
)
```

### Multiple Objects
```python
multi_process(
    process,
    items,
    backend='ray',
    shared_kwargs=['model', 'lookup', 'embeddings'],
    model=model,
    lookup=lookup_table,
    embeddings=embeddings
)
```

### With **kwargs
```python
def process(id, **kwargs):
    model = kwargs['model']
    return model[id % 100]

multi_process(
    process,
    items,
    backend='ray',
    shared_kwargs=['model'],  # Works!
    model=large_model
)
```

## When to Use

✅ **Use** when:
- Objects > 1 MB
- Multiple workers
- Many tasks
- Read-only data
- NumPy arrays

❌ **Skip** when:
- Small objects (< 1 MB)
- Single worker
- Few tasks
- Frequently modified data

## Common Errors

```python
# ❌ Key not in kwargs
shared_kwargs=['model']  # But no model=... provided

# ❌ Invalid parameter name  
shared_kwargs=['nonexistent']  # my_func doesn't have this param

# ❌ Wrong backend
backend='mp', shared_kwargs=['model']  # Must use 'ray'
```

All validated automatically with clear error messages!

## Performance Tips

1. **Bigger = Better**: More benefit with larger objects
2. **NumPy Preferred**: True zero-copy for NumPy arrays
3. **Share Selectively**: Don't share everything, only large objects
4. **Monitor Memory**: Ray object store has limits

## See Also

- [Complete Documentation](./zero_copy_sharing.md)
- [Examples](../examples/shared_kwargs_example.py)
- [Tests](../tests/test_shared_kwargs.py)
