from speedy_utils import multi_process
import numpy as np
import ray
import time
import torch

if __name__ == '__main__':

    def f(i, data=None):
        print(f"Task {i}")
        # task_id = ray.get_runtime_context().get_task_id()
        # tensor_data = torch.tensor(data[i])
        time.sleep(1)

    inputs = np.random.rand(100*10).reshape(100, 10)
    multi_process(f, range(100), data=inputs, shared_kwargs=['data'], backend='ray', workers=4)