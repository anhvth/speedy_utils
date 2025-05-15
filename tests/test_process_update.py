"""
Tests for the updated multi_process functionality with process_update_interval.
"""
import time
import pytest
import multiprocessing
from unittest.mock import patch, MagicMock
from speedy_utils.multi_worker.process import multi_process

# Test helper functions - must be at module level for pickling
def identity(x):
    """Simple identity function for testing."""
    return x

def slow_identity(x, delay=0.01):
    """Identity function with a slight delay to test progress updates."""
    time.sleep(delay)
    return x

def failing_function(x):
    """Function that raises an error for a specific input."""
    if x == 5:
        raise ValueError("Test error")
    return x

def test_process_update_interval():
    """Test that the process_update_interval parameter works correctly."""
    # Create a list of 20 items to process
    test_input = list(range(20))
    
    # Mock tqdm to check if updates are performed correctly
    with patch('speedy_utils.multi_worker.process.tqdm') as mock_tqdm:
        # Setup a mock progress bar
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar
        
        # Run multi_process with progress=True and process_update_interval=5
        result = multi_process(
            slow_identity, 
            test_input, 
            workers=2,
            progress=True,
            process_update_interval=5
        )
        
        # Check results
        assert result == test_input
        
        # Verify tqdm was called
        mock_tqdm.assert_called_once()
        
        # Verify the bar's update method was called
        assert mock_bar.update.call_count > 0
        
        # Verify the bar was closed
        mock_bar.close.assert_called_once()

def test_worker_error_handling():
    """Test error handling in the worker process."""
    # Since one worker might process multiple items in batch,
    # and we don't know the exact order, we'll make a more robust test
    result = multi_process(
        failing_function,
        range(10),
        stop_on_error=False
    )
    
    # Count results
    none_count = result.count(None)
    assert none_count > 0, "Expected at least one None in results due to error"
    
    # Check total length
    assert len(result) == 10, "Expected 10 results in total"
    
    # Check counts of valid numbers
    valid_numbers = [x for x in result if x is not None]
    assert len(valid_numbers) == 9, "Expected 9 valid numbers"
    
    for i in range(10):
        if i != 5:
            assert i in result, f"Expected {i} in results"

def test_batch_parameter():
    """Test the batch parameter for multi_process."""
    # Create a list of 20 items to process
    test_input = list(range(20))
    
    # Process with batch=5
    result = multi_process(
        identity,
        test_input,
        batch=5
    )
    
    # Check results
    assert result == test_input
