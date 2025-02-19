import time
from fastcore.all import threaded
from loguru import logger
from tqdm import tqdm
from multiprocessing import Manager

def multi_process(
    func,
    inputs,
    workers=4,
    verbose=True,
    error_is_none=True,
    stop_on_error=True,
):
    from speedy_utils.common.utils_cache import identify

    manager = Manager()
    errors = manager.list()
    shared_results = manager.dict()
    share_count = manager.Value("i", 0)

    @threaded(process=True)
    def f_wrapper(item, i_id):
        try:
            result = func(item)
        except Exception as e:
            errors.append(e)
            if error_is_none:
                result = None
            else:
                logger.error(f"Error with input {item}: {e}")
                result = str(e)

        shared_results[i_id] = result
        share_count.value += 1
        logger.debug(f"Processed {share_count.value}/{len(inputs)}")

    for i, item in enumerate(inputs):
        f_wrapper(item, i)

    

    return results

if __name__ == "__main__":

    def f(x):
        time.sleep(0.1)
        return x + 1

    class Aclass:
        def f(self, x, y):
            time.sleep(1)
            return x + y

    obj = Aclass()
    inputs = [(i, i + 1) for i in range(10)]

    def f2(x):
        return obj.f(x[0], x[1])

    f3 = lambda x: obj.f(x[0], x[1])

    results = multi_process(
        f3,
        inputs,
        workers=2,
        verbose=True,
        stop_on_error=False,
        error_is_none=False,
    )
    logger.success(f"Results: {results}")

