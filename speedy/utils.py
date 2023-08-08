import concurrent.futures
import inspect
import json
from multiprocessing import Pool
import os
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import xxhash
from loguru import logger
from tqdm import tqdm

AV_CACHE_DIR = osp.join(osp.expanduser('~'), '.cache/av')
ICACHE = dict()


def mkdir_or_exist(dir_name):
    return os.makedirs(dir_name, exist_ok=True)


def dump_json_or_pickle(obj, fname):
    """
        Dump an object to a file, support both json and pickle
    """
    if fname.endswith('.json'):
        with open(fname, 'w') as f:
            json.dump(obj, f)
    else:
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)


def load_json_or_pickle(fname):
    """
        Load an object from a file, support both json and pickle
    """
    if fname.endswith('.json'):
        with open(fname, 'r') as f:
            return json.load(f)
    else:
        with open(fname, 'rb') as f:
            return pickle.load(f)


def identify(x):
    '''Return an hex digest of the input'''
    return xxhash.xxh64(pickle.dumps(x), seed=0).hexdigest()


def memoize(func):
    '''Cache result of function call on disk
    Support multiple positional and keyword arguments'''
    @wraps(func)
    def memoized_func(*args, **kwargs):
        try:
            if 'cache_key' in kwargs:
                cache_key = kwargs['cache_key']

                func_id = identify((inspect.getsource(func))) + \
                    '_cache_key_'+str(kwargs['cache_key'])
            else:
                func_id = identify((inspect.getsource(func), args, kwargs))
            cache_path = os.path.join(
                AV_CACHE_DIR, 'funcs', func.__name__+'/'+func_id)
            mkdir_or_exist(os.path.dirname(cache_path))

            if (os.path.exists(cache_path) and
                    not func.__name__ in os.environ and
                    not 'BUST_CACHE' in os.environ):
                result = pickle.load(open(cache_path, 'rb'))
            else:
                result = func(*args, **kwargs)
                pickle.dump(result, open(cache_path, 'wb'))
            return result
        except (KeyError, AttributeError, TypeError, Exception) as e:
            logger.warning(f'Exception: {e}, use default function call')
            return func(*args, **kwargs)
    return memoized_func


def imemoize(func):
    """
        Memoize a function into memory, the function recaculate only 
        change when its belonging arguments change
    """
    @wraps(func)
    def _f(*args, **kwargs):
        ident_name = identify((inspect.getsource(func), args, kwargs))
        try:
            result = ICACHE[ident_name]
        except:
            result = func(*args, **kwargs)
            ICACHE[ident_name] = result
        return result
    return _f


def multi_thread(func, args_list, kwargs_list=None, pbar='tqdm', num_workers=4):
    """
    Executes a function in parallel using multiple threads.

    Parameters:
    - func: The function to execute.
    - args_list: A list of argument tuples. Each tuple contains arguments for one call of `func`.
    - kwargs_list: A list of dictionaries. Each dictionary contains keyword arguments for one call of `func`.
    - pbar: Whether to show a progress bar. Default is 'tqdm'. If None, no progress bar is shown.
    - num_workers: The number of worker threads.

    Returns:
    A list of results.
    """

    # If kwargs_list is not provided or is None, make it an empty dict for each set of args.
    if kwargs_list is None:
        kwargs_list = [{} for _ in args_list]

    # Function to execute in each thread
    def wrapper(args, kwargs):
        return func(*args, **kwargs)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        if pbar:
            for result in tqdm(executor.map(wrapper, args_list, kwargs_list), total=len(args_list)):
                results.append(result)
        else:
            for result in executor.map(wrapper, args_list, kwargs_list):
                results.append(result)

    return results


def multi_process(f, inputs, max_workers=8, desc='',
               unit='Samples', verbose=True, pbar_iterval=10):
    if verbose:
        
        logger.info('Multi processing {} | Num samples: {}', f.__name__, len(inputs))
        pbar = tqdm(total=len(inputs))
        
    with Pool(max_workers) as p:
        it = p.imap(f, inputs)
        return_list = []
        for i, ret in enumerate(it):
            return_list.append(ret)
            if i % pbar_iterval == 0 and verbose:
                pbar.update(pbar_iterval)
    return return_list