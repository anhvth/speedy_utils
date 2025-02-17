from speedy_utils.all import *

setup_logger("D")


def slow_func(x):
    time.sleep(0.1)  # sleeps for `x` seconds
    if x == 5:
        raise ValueError("AAAAAAAAAAAAA")
    return x


results = multi_thread_in_sub_process(
    func=slow_func,
    orig_inputs=range(10),
    workers=2,
    desc="Demo",
)


results
