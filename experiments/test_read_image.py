# type: ignore
import ray
from tqdm import tqdm


ray.init()
import os
import random
import time

import cv2


@ray.remote
def read_image(i: int):
    img = cv2.imread('experiments/exp1/test.png')
    my_hostname = os.uname()[1]
    img_resized = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    t = random.uniform(0.1, 2)
    time.sleep(t)
    return i, img_resized, my_hostname


def process_results(futures):
    remaining = futures
    for _ in range(len(futures)):
        ready, remaining = ray.wait(remaining)
        # yeild i, img, hostname
        i, img, hostname = ray.get(ready[0])
        yield i, img, hostname


n_iters = 100
futures = [read_image.remote(i) for i in range(n_iters)]

for i, img, hostname in tqdm(process_results(futures), total=n_iters):
    print(f'Iteration {i}: Image shape: {img.shape}, Hostname: {hostname}')
