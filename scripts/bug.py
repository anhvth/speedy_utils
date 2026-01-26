import llm_utils
from speedy_utils import multi_process

llm = llm_utils.LLM(client=8666, timeout=3)

def f(x):
    return llm('hello world')


# multi_process(f, range(10), backend='mp', error_handler='ignore')

f(0)