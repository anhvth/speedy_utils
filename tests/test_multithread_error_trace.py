# type: ignore
"""
Comprehensive demonstration of improved multi_thread error tracing.

BEFORE: Errors showed lots of multi_thread/concurrent.futures infrastructure
AFTER: Errors focus on user code with clear, actionable messages
"""

from speedy_utils import multi_thread


def demo_1_simple_error():
    """Demo 1: Simple function error."""
    print('\n' + '=' * 70)
    print('DEMO 1: Simple TypeError in user function')
    print('=' * 70)

    def process_item(x):
        # Intentional error - calling a list
        my_list = [1, 2, 3]
        return my_list(x)

    try:
        multi_thread(process_item, range(3), workers=2, progress=False)
    except TypeError as e:
        print(f'\nCaught: {type(e).__name__}')
        print('\nError message focuses on YOUR code:')
        print(f'{e}')


def demo_2_nested_functions():
    """Demo 2: Error in nested function calls."""
    print('\n\n' + '=' * 70)
    print('DEMO 2: ZeroDivisionError in nested helper function')
    print('=' * 70)

    def process_data(val):
        def validate_and_calc(x):
            # Validation
            if x < 0:
                raise ValueError(f'Negative not allowed: {x}')
            # Calculation that might fail
            return 100 / x  # Fails when x=0

        return validate_and_calc(val)

    try:
        multi_thread(process_data, [1, 2, 0, 3], workers=2, progress=False)
    except ZeroDivisionError as e:
        print(f'\nCaught: {type(e).__name__}')
        print('\nError trace shows YOUR function call chain:')
        print(f'{e}')


def demo_3_wrong_parameter_type():
    """Demo 3: Passing wrong type to multi_thread."""
    print('\n\n' + '=' * 70)
    print('DEMO 3: Common mistake - passing list instead of function')
    print('=' * 70)

    # Create a list of functions
    functions = [
        lambda x: x * 2,
        lambda x: x + 10,
        lambda x: x**2,
    ]

    try:
        # WRONG: passing list as func parameter
        # Should be: multi_thread(some_func, functions)
        multi_thread(functions, range(3), workers=2, progress=False)
    except TypeError as e:
        print(f'\nCaught: {type(e).__name__}')
        print('\nHelpful error message:')
        print(f'{e}')


def demo_4_attribute_error():
    """Demo 4: AttributeError in user code."""
    print('\n\n' + '=' * 70)
    print('DEMO 4: AttributeError accessing non-existent attribute')
    print('=' * 70)

    def process_dict(data):
        # Trying to access non-existent attribute
        return data.nonexistent_method()

    try:
        items = [{'key': 'value'}, {'key2': 'value2'}]
        multi_thread(process_dict, items, workers=2, progress=False)
    except AttributeError as e:
        print(f'\nCaught: {type(e).__name__}')
        print('\nClear error pointing to YOUR code:')
        print(f'{e}')


if __name__ == '__main__':
    print('=' * 70)
    print('IMPROVED multi_thread ERROR TRACING DEMONSTRATIONS')
    print('=' * 70)
    print('\nKey improvements:')
    print('  ✓ Errors show function name and problematic input')
    print('  ✓ Tracebacks filtered to show only user code')
    print('  ✓ No concurrent.futures/threading clutter')
    print('  ✓ Helpful hints for common mistakes')
    print('  ✓ Clear, actionable error messages')

    demo_1_simple_error()
    demo_2_nested_functions()
    demo_3_wrong_parameter_type()
    demo_4_attribute_error()

    print('\n\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print('All errors now focus on YOUR code, making debugging')
    print('significantly faster and easier!')
    print('=' * 70)
