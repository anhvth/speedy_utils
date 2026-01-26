from speedy_utils import *

def do_something(x):
    x = 10
    y = 0
    x/y

if __name__ == '__main__':
    inputs = range(10)
    
    print("=" * 80)
    print("Testing MP backend:")
    print("=" * 80)
    try:
        multi_process(do_something, inputs, backend='mp')
    except SystemExit:
        pass
    
    print("\n" + "=" * 80)
    print("Testing RAY backend:")
    print("=" * 80)
    try:
        multi_process(do_something, inputs, backend='ray')
    except SystemExit:
        pass
