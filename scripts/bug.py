# type: ignore
from speedy_utils import multi_process, multi_thread


def do_something(x):
    if x % 3 == 0:
        raise ValueError(f'Error at index {x}')
    return x * 2


inputs = range(10)


if __name__ == '__main__':
    print('Testing error_handler="log" with mp backend:')
    results = multi_process(
        do_something,
        inputs,
        backend='mp',
        error_handler='log',
        max_error_files=5,
    )
    print(f'Results: {results}')
    print()

    # print('Testing error_handler="log" with multi_thread:')
    # results = multi_thread(
    #     do_something,
    #     inputs,
    #     error_handler='log',
    #     max_error_files=5,
    # )
    # print(f'Results: {results}')

