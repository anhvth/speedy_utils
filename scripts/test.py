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

    # try:
    multi_thread(process_item, range(300), workers=100, progress=True)
    # except TypeError as e:
    #     pass
        # import traceback; traceback.print_exc()
        # print(f'\nCaught: {type(e).__name__}')
        # print('\nError message focuses on YOUR code:')
        # print(f'{e}')



demo_1_simple_error()