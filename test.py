from speedy_utils.multi_worker.thread import multi_thread


def f(x):
    x = x['x']
    print(x)
    return 

item = [{'x': i} for i in range(5)]
multi_thread(f, item)