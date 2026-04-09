from speedy_utils import *


def do_something(x):
    x = 10
    y = 0
    x/y  # type: ignore[operator]


do_something(1)

