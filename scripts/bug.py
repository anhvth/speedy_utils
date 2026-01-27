import llm_utils
from speedy_utils import multi_process
from fastcore.all import call_parse

def f(x):
    y = 1
    if x%3 == 0:
        1/0
    return 


def test(backend) -> None:
    multi_process(f, range(1000), backend=backend, error_handler='log')

test('ray')
test('mp')