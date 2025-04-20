from speedy_utils.all import *


def f_single(x):
    if x == 2:
        raise ValueError("Error")
    return x + 1


def f_tuple(x, y):
    return x + y


def f_dict(x, y=-1000):
    return x + y

if __name__ == "__main__":
    input_sing = [1, 2, 3, 4, 5]
    # multi_thread(f_single, input_sing, workers=2, input_type="single")

    input_tuple = [(1, 2), (3, 4), (5, 6)]
    multi_thread(f_tuple, input_tuple, workers=2, input_type="tuple")

    input_dict = [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5}]
    multi_thread(f_dict, input_dict, workers=2, input_type="dict")
# def f_dict(x,y,z=0):
#     return x+y+z
