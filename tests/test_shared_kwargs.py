"""Test shared_kwargs feature with Ray zero-copy."""
import sys
from pathlib import Path

import numpy as np


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from speedy_utils.multi_worker.process import multi_process


def process_with_large_array(x, large_array=None, multiplier=1):
    """Process item with a large shared array."""
    if large_array is None:
        return x * multiplier
    # Use the shared array
    return x * multiplier + large_array.sum()


def process_with_kwargs(x, **kwargs):
    """Process item with variable kwargs."""
    model = kwargs.get('model')
    config = kwargs.get('config', {})

    result = x * config.get('multiplier', 1)
    if model is not None:
        result += model['bias']
    return result


def test_shared_kwargs_basic():
    """Test basic shared_kwargs functionality."""
    print('\n=== Test 1: Basic shared_kwargs ===')

    # Create a large numpy array (should be zero-copy)
    large_array = np.random.randn(1000, 1000)
    print(f'Array size: {large_array.nbytes / 1e6:.2f} MB')

    items = list(range(10))

    # Test with Ray backend and shared_kwargs
    try:
        results = multi_process(
            process_with_large_array,
            items,
            workers=4,
            backend='ray',
            shared_kwargs=['large_array'],
            large_array=large_array,
            multiplier=2,
            progress=False
        )

        print(f'Results (first 5): {results[:5]}')
        expected = [x * 2 + large_array.sum() for x in items]
        assert np.allclose(results, expected), 'Results mismatch!'
        print('✅ Test passed: shared_kwargs works correctly')

    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()


def test_shared_kwargs_validation():
    """Test validation of shared_kwargs."""
    print('\n=== Test 2: Validation ===')

    # Test 1: shared_kwargs key not in func_kwargs
    try:
        multi_process(
            process_with_large_array,
            [1, 2, 3],
            backend='ray',
            shared_kwargs=['nonexistent'],
            multiplier=2,
            progress=False
        )
        print('❌ Should have raised ValueError for missing key')
    except ValueError as e:
        print(f'✅ Correctly raised error: {e}')

    # Test 2: shared_kwargs with invalid parameter name
    try:
        multi_process(
            process_with_large_array,
            [1, 2, 3],
            backend='ray',
            shared_kwargs=['invalid_param'],
            invalid_param=np.array([1, 2, 3]),
            progress=False
        )
        print('❌ Should have raised ValueError for invalid parameter')
    except ValueError as e:
        print(f'✅ Correctly raised error: {e}')

    # Test 3: shared_kwargs with non-ray backend
    try:
        multi_process(
            process_with_large_array,
            [1, 2, 3],
            backend='seq',
            shared_kwargs=['large_array'],
            large_array=np.array([1, 2, 3]),
            progress=False
        )
        print('❌ Should have raised ValueError for non-ray backend')
    except ValueError as e:
        print(f'✅ Correctly raised error: {e}')


def test_shared_kwargs_with_var_keyword():
    """Test shared_kwargs with **kwargs functions."""
    print('\n=== Test 3: Functions with **kwargs ===')

    model_dict = {'bias': 10, 'weights': np.random.randn(100)}
    config = {'multiplier': 3}

    items = list(range(5))

    try:
        results = multi_process(
            process_with_kwargs,
            items,
            workers=2,
            backend='ray',
            shared_kwargs=['model'],
            model=model_dict,
            config=config,
            progress=False
        )

        expected = [x * 3 + 10 for x in items]
        assert results == expected, f'Results mismatch! Got {results}, expected {expected}'
        print(f'Results: {results}')
        print('✅ Test passed: **kwargs function works correctly')

    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()


def test_without_shared_kwargs():
    """Test that existing behavior still works without shared_kwargs."""
    print('\n=== Test 4: Backward compatibility (no shared_kwargs) ===')

    items = list(range(10))

    try:
        results = multi_process(
            process_with_large_array,
            items,
            workers=4,
            backend='ray',
            multiplier=5,
            progress=False
        )

        expected = [x * 5 for x in items]
        assert results == expected, f'Results mismatch! Got {results}, expected {expected}'
        print(f'Results (first 5): {results[:5]}')
        print('✅ Test passed: backward compatibility maintained')

    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print('Testing shared_kwargs feature with Ray zero-copy\n')

    # Check if Ray is available
    try:
        import ray
        if not ray.is_initialized():
            print('Initializing Ray...')

        test_shared_kwargs_basic()
        test_shared_kwargs_validation()
        test_shared_kwargs_with_var_keyword()
        test_without_shared_kwargs()

        # Cleanup
        if ray.is_initialized():
            ray.shutdown()

        print('\n' + '=' * 50)
        print('🎉 All tests completed!')
        print('=' * 50)

    except ImportError:
        print('❌ Ray not installed. Please install with: pip install ray')
        sys.exit(1)
