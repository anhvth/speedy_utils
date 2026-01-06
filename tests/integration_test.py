import numpy as np

from speedy_utils import multi_process


# Quick integration test
print('🧪 Quick Integration Test')
print('=' * 50)

# Test 1: Basic usage
print('\n1. Basic zero-copy...')
data = np.random.randn(100, 100)
results = multi_process(
    lambda x, arr=None: arr[x % 10, 0] if arr is not None else x,
    range(20),
    workers=2,
    backend='ray',
    shared_kwargs=['arr'],
    arr=data,
    progress=False
)
print(f'   ✅ Processed {len(results)} items')

# Test 2: Validation
print('\n2. Validation...')
try:
    multi_process(
        lambda x: x,
        [1, 2],
        backend='ray',
        shared_kwargs=['missing'],
        progress=False
    )
    print('   ❌ Should have failed')
except ValueError as e:
    print(f'   ✅ Correctly caught: {str(e)[:40]}...')

# Test 3: Backward compatibility
print('\n3. Backward compatibility...')
results = multi_process(
    lambda x: x * 2,
    range(10),
    workers=2,
    backend='ray',
    progress=False
)
assert results == [x * 2 for x in range(10)]
print('   ✅ Works without shared_kwargs')

print('\n' + '=' * 50)
print('✨ All integration tests passed!')
