from speedy_utils import *

def do_something(x):
    x = 10
    y = 0
    x/y

if __name__ == '__main__':
    inputs = range(10)
    multi_process(do_something, inputs, backend='ray')
