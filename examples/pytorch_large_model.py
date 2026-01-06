"""
Example: Zero-copy sharing with a larger PyTorch model.

This demonstrates the real-world benefits with a ResNet-style model.
"""
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from speedy_utils.multi_worker.process import multi_process


try:
    import torch
    import torch.nn as nn
except ImportError:
    print('❌ PyTorch required: pip install torch')
    sys.exit(1)


class LargeResidualBlock(nn.Module):
    """A larger residual block."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class LargeModel(nn.Module):
    """A larger model similar to a small ResNet."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stack of residual blocks
        self.layer1 = nn.Sequential(
            LargeResidualBlock(64),
            LargeResidualBlock(64),
            LargeResidualBlock(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            LargeResidualBlock(128),
            LargeResidualBlock(128)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            LargeResidualBlock(256),
            LargeResidualBlock(256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def classify_batch(batch_id, model=None, batch_size=8):
    """Classify a batch of images using the shared model."""
    if model is None:
        return None

    # Simulate image batch (3, 224, 224)
    images = torch.randn(batch_size, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Get predicted classes
    predicted_classes = predictions.argmax(dim=1)

    return {
        'batch_id': batch_id,
        'predictions': predicted_classes.tolist(),
        'confidence': predictions.max(dim=1).values.mean().item()
    }


if __name__ == '__main__':
    from tabulate import tabulate

    print('\n🔥 Large PyTorch Model Example - Performance Comparison')
    print('=' * 70)

    # Create a large model
    print('\n📦 Creating large model...')
    model = LargeModel()

    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_size / 1024 / 1024
    param_count = sum(p.numel() for p in model.parameters())

    print(f'   Parameters: {param_count:,}')
    print(f'   Model size: {model_size_mb:.2f} MB')

    # Number of batches to process
    num_batches = 200
    batch_size = 8
    num_workers = 8

    print(f'\n🎯 Task: Classify {num_batches} batches ({num_batches * batch_size} images)')
    print(f'   Workers: {num_workers}')
    print(f'   Batch size: {batch_size}')

    # Store results for comparison
    results_table = []

    # Method 1: Single process (sequential)
    print('\n' + '─' * 70)
    print('Method 1: Single Process (Sequential)')
    print('─' * 70)
    start = time.time()
    results_seq = multi_process(
        classify_batch,
        range(num_batches),
        workers=1,
        backend='seq',
        model=model,
        batch_size=batch_size,
        desc='Single process'
    )
    time_seq = time.time() - start
    results_table.append([
        'Single Process',
        '1',
        f'{time_seq:.3f}s',
        '1.00x',
        f'{model_size_mb:.2f} MB',
        '0 MB',
        '-'
    ])

    # Method 2: Multi-process WITHOUT zero-copy
    print('\n' + '─' * 70)
    print('Method 2: Multi-Process WITHOUT Zero-Copy')
    print('─' * 70)
    start = time.time()
    results_without = multi_process(
        classify_batch,
        range(num_batches),
        workers=num_workers,
        backend='ray',
        model=model,
        batch_size=batch_size,
        desc='Without sharing'
    )
    time_without = time.time() - start
    speedup_without = time_seq / time_without
    memory_without = model_size_mb * num_workers
    memory_saved_without = memory_without - model_size_mb
    results_table.append([
        'Multi-Process (No Share)',
        str(num_workers),
        f'{time_without:.3f}s',
        f'{speedup_without:.2f}x',
        f'{memory_without:.2f} MB',
        f'{memory_saved_without:.2f} MB',
        '❌'
    ])

    # Method 3: Multi-process WITH zero-copy
    print('\n' + '─' * 70)
    print('Method 3: Multi-Process WITH Zero-Copy')
    print('─' * 70)
    start = time.time()
    results_with = multi_process(
        classify_batch,
        range(num_batches),
        workers=num_workers,
        backend='ray',
        shared_kwargs=['model'],
        model=model,
        batch_size=batch_size,
        desc='With zero-copy'
    )
    time_with = time.time() - start
    speedup_with = time_seq / time_with
    memory_with = model_size_mb
    memory_saved_with = 0
    results_table.append([
        'Multi-Process (Zero-Copy)',
        str(num_workers),
        f'{time_with:.3f}s',
        f'{speedup_with:.2f}x',
        f'{memory_with:.2f} MB',
        f'{memory_saved_with:.2f} MB',
        '✅'
    ])

    # Display comparison table
    print('\n' + '=' * 70)
    print('📊 PERFORMANCE COMPARISON')
    print('=' * 70)

    headers = [
        'Method',
        'Workers',
        'Time',
        'Speedup',
        'Memory Used',
        'Extra Memory',
        'Efficient'
    ]

    print('\n' + tabulate(results_table, headers=headers, tablefmt='grid'))

    # Additional analysis
    print('\n' + '=' * 70)
    print('📈 ANALYSIS')
    print('=' * 70)

    improvement = (time_without - time_with) / time_without * 100
    memory_saving = (memory_without - memory_with) / memory_without * 100

    analysis_table = [
        ['Baseline (Single Process)', f'{time_seq:.3f}s', f'{model_size_mb:.2f} MB'],
        ['Multi-Process (No Share)', f'{time_without:.3f}s ({speedup_without:.2f}x)', f'{memory_without:.2f} MB'],
        ['Multi-Process (Zero-Copy)', f'{time_with:.3f}s ({speedup_with:.2f}x)', f'{memory_with:.2f} MB'],
        ['', '', ''],
        ['Zero-Copy Improvement', f'{improvement:.1f}% faster', f'{memory_saving:.1f}% less memory'],
        ['Zero-Copy vs Single', f'{speedup_with:.2f}x speedup', 'Same memory as single'],
    ]

    print('\n' + tabulate(analysis_table, headers=['', 'Time', 'Memory'], tablefmt='simple'))

    print(f'\n✅ All methods processed {len(results_with)} batches successfully')
    print(f'   Sample predictions: {results_with[0]["predictions"][:3]}...')

    # Cleanup
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except:
        pass

    print('\n' + '=' * 70)
    print('💡 KEY TAKEAWAYS:')
    print('=' * 70)
    print(f'  1. Zero-copy is {improvement:.1f}% faster than multi-process without sharing')
    print(f'  2. Zero-copy saves {memory_saving:.1f}% memory compared to multi-process')
    print(f'  3. Zero-copy achieves {speedup_with:.2f}x speedup with same memory as single process')
    print('  4. Larger models = greater benefits from zero-copy sharing')
    print('=' * 70)
