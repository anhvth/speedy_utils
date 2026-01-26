"""Test improved error handling in multi_thread."""
from speedy_utils import multi_thread


def process_data(item):
    """Process a data item - may raise errors."""
    # Simulate some processing logic
    value = item['value']
    threshold = item.get('threshold', 100)
    
    # This will error when value == threshold
    result = 1 / (value - threshold)
    return result * 2


def main():
    """Run test cases for error handling."""
    # Test case 1: Error in the middle of processing
    print("Test 1: Error with division by zero")
    print("-" * 60)
    
    data = [
        {'value': 50, 'threshold': 100},
        {'value': 75, 'threshold': 100},
        {'value': 100, 'threshold': 100},  # This will error
        {'value': 125, 'threshold': 100},
    ]
    
    try:
        results = multi_thread(process_data, data, workers=4)
        print(f"Results: {results}")
    except SystemExit:
        print("\nTest completed - error was properly reported")


if __name__ == '__main__':
    main()
