"""
Example demonstrating Ray zero-copy sharing of large objects.

This example shows how to use shared_kwargs to efficiently share large
objects (like models, datasets, or configuration) across workers without
duplicating memory.
"""
import sys
import time
from pathlib import Path

import numpy as np


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from speedy_utils.multi_worker.process import multi_process


def process_with_model(item_id, model=None, config=None):
    """
    Simulate processing with a large model.
    
    In real-world scenarios:
    - model could be a large ML model (PyTorch, TensorFlow, etc.)
    - config could be shared configuration dictionaries
    - Large numpy arrays for lookup tables, embeddings, etc.
    """
    # Simulate some work with the model
    if model is not None:
        weights = model['weights']
        bias = model['bias']
        # Use model for inference (just access some values)
        result = weights[item_id % len(weights), 0] + bias[0] + item_id
    else:
        result = item_id

    return result


def example_1_basic_sharing():
    """Example 1: Basic shared_kwargs usage."""
    print('\n' + '=' * 60)
    print('Example 1: Basic Zero-Copy Sharing')
    print('=' * 60)

    # Create a large model (simulating a real model)
    model = {
        'weights': np.random.randn(1000, 1000),  # 8 MB
        'bias': np.array([1.5])
    }

    print(f'Model size: {model["weights"].nbytes / 1e6:.2f} MB')
    print('Processing 100 items with 4 workers...\n')

    items = list(range(100))

    # Without shared_kwargs - model is serialized/copied for each task
    start = time.time()
    results_without_sharing = multi_process(
        process_with_model,
        items,
        workers=4,
        backend='ray',
        model=model,  # This will be copied/serialized
        desc='Without sharing'
    )
    time_without = time.time() - start

    # With shared_kwargs - model is put in Ray's object store (zero-copy)
    start = time.time()
    results_with_sharing = multi_process(
        process_with_model,
        items,
        workers=4,
        backend='ray',
        shared_kwargs=['model'],  # Zero-copy via Ray object store
        model=model,
        desc='With zero-copy'
    )
    time_with = time.time() - start

    print(f'\n⏱️  Time without sharing: {time_without:.3f}s')
    print(f'⏱️  Time with zero-copy: {time_with:.3f}s')
    print(f'🚀 Speedup: {time_without / time_with:.2f}x')

    # Verify results are identical
    assert np.allclose(results_without_sharing, results_with_sharing)
    print('✅ Results are identical!')


def example_2_multiple_shared_objects():
    """Example 2: Sharing multiple large objects."""
    print('\n' + '=' * 60)
    print('Example 2: Sharing Multiple Large Objects')
    print('=' * 60)

    # Multiple large objects to share
    model = {
        'weights': np.random.randn(500, 500),
        'bias': np.array([2.0])
    }
    lookup_table = np.random.randn(1000, 100)  # Large lookup table
    config = {'threshold': 0.5, 'normalize': True}

    print(f'Model size: {model["weights"].nbytes / 1e6:.2f} MB')
    print(f'Lookup table size: {lookup_table.nbytes / 1e6:.2f} MB')

    def process_with_multiple_objects(idx, model=None, lookup=None, config=None):
        if model is None or lookup is None:
            return idx
        result = model['weights'][0, 0] + lookup[idx % len(lookup), 0]
        if config and config.get('normalize'):
            result = result / 2.0
        return result + idx

    items = list(range(50))

    # Share all large objects via zero-copy
    results = multi_process(
        process_with_multiple_objects,
        items,
        workers=4,
        backend='ray',
        shared_kwargs=['model', 'lookup'],  # Share both objects
        model=model,
        lookup=lookup_table,
        config=config,  # Small config - no need to share
        desc='Multi-object sharing'
    )

    print(f'✅ Processed {len(results)} items with zero-copy sharing')


def example_3_memory_efficiency():
    """Example 3: Demonstrate memory efficiency."""
    print('\n' + '=' * 60)
    print('Example 3: Memory Efficiency Comparison')
    print('=' * 60)

    # Create a very large object
    large_dataset = np.random.randn(2000, 2000)  # ~32 MB
    print(f'Large dataset size: {large_dataset.nbytes / 1e6:.2f} MB')

    def simple_process(idx, dataset=None):
        if dataset is None:
            return idx
        # Just access a small part of the dataset
        return dataset[idx % 100, idx % 100] + idx

    items = list(range(20))

    print('\n📊 Memory usage analysis:')
    print('- Without sharing: Each worker gets a copy (~32 MB × 4 = 128 MB)')
    print('- With zero-copy: Single shared object in object store (32 MB)')
    print('- Memory saved: ~96 MB')

    results = multi_process(
        simple_process,
        items,
        workers=4,
        backend='ray',
        shared_kwargs=['dataset'],
        dataset=large_dataset,
        desc='Zero-copy large dataset'
    )

    print(f'✅ Processed {len(results)} items efficiently')


if __name__ == '__main__':
    print('\n🚀 Ray Zero-Copy Sharing Examples')
    print('=' * 60)

    try:
        import ray

        # Run examples
        example_1_basic_sharing()
        example_2_multiple_shared_objects()
        example_3_memory_efficiency()

        # Cleanup
        if ray.is_initialized():
            ray.shutdown()

        print('\n' + '=' * 60)
        print('✨ All examples completed successfully!')
        print('=' * 60)
        print('\n📖 Key takeaways:')
        print('  1. Use shared_kwargs for large objects to save memory')
        print('  2. Zero-copy works best with numpy arrays')
        print('  3. Share model weights, datasets, lookup tables, etc.')
        print('  4. Significant speedup for large object sizes')
        print('=' * 60)

    except ImportError:
        print('❌ Ray not installed. Please install with: pip install ray')
        sys.exit(1)
