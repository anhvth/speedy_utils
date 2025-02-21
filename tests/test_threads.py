import time
from speedy_utils.all import *
from speedy_utils.common.clock import Clock
from multiprocessing import freeze_support

def run_test():
    def f(x):
        time.sleep(0.1)
        return x + 1

    class Aclass:
        def f(self, x, y=None):
            time.sleep(0.1)
            if x % 10 == 0:
                raise ValueError(f"Error with {x}")
            return x

    obj = Aclass()
    # Test with tuple inputs
    inputs_tuple = [(i, i + 1) for i in range(30)]
    
    # Test with dict inputs
    inputs_dict = [{"x": i, "y": i + 1} for i in range(30, 60)]
    input_numbers = [i for i in range(30)]
    def f2(x, y):
        return obj.f(x, y)

    # Test both input types
    results_tuple = multi_thread(f2, inputs_tuple, workers=10, verbose=True, process=False, desc="Tuple")
    results_dict = multi_thread(f2, inputs_dict, workers=10, verbose=True, process=False, desc="Dict")
    results_numbers = multi_thread(f, input_numbers, workers=10, verbose=True, process=False, desc="Numbers")
    
    return results_tuple, results_dict

if __name__ == '__main__':
    freeze_support()
    run_test()

