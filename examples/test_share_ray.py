from speedy_utils import *
import numpy as np
import ray
import time
import torch

if __name__ == '__main__':

    def f(i, data=None):
        return data+i
        
    inputs = np.random.rand(300000*1000).reshape(-1, 10)
    outputs = multi_process(f, range(1000), 
        data=inputs, 
        shared_kwargs=['data'],
        backend='ray', 
        workers=16,
        lazy_output=True, 
        dump_in_thread=True)
    import gc; gc.collect()
    print('Done, sleep 10')

    time.sleep(10)
    