from speedy_utils import *
from speedy_utils.all import *


def f(x):
    x = x['x']
    print(x)
    return 

item = [{'x': i} for i in range(5)]
multi_thread(f, item)