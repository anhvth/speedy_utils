from typing import Any, Callable, Dict, List, Union

import pandas as pd
from loguru import logger

from speedy_utils.common.clock import Clock

clock = Clock()

# def handle_inputs(func: Callable) -> Callable:
#     """
#     Decorator to handle different input formats for the multi_process function.
#     """

#     def wrapper(f: Callable, inputs: Union[List[Dict[str, Any]], List[Any], pd.DataFrame], *args, **kwargs):
#         if f.__code__.co_argcount == 1:
#             if isinstance(inputs, pd.DataFrame):
#                 inputs = inputs.to_dict("records")
#             logger.debug("Function has only 1 argument, converting list of dictionaries to list of values...")
#             assert isinstance(inputs, list), "inputs must be a list"
#             arg_name = f.__code__.co_varnames[0]
#             # if isinstance(inputs[0], dict):
#             inputs = [{arg_name: input_} for input_ in inputs]
#             return func(f, inputs, *args, **kwargs)
#         else:
#             if isinstance(inputs, pd.DataFrame):
#                 logger.debug("Converting DataFrame to list of dictionaries...")
#                 inputs = inputs.to_dict("records")
#             elif isinstance(inputs[0], (list, tuple)):
#                 args_names = f.__code__.co_varnames[: f.__code__.co_argcount]
#                 inputs = [{argname: i for argname, i in zip(args_names, item)} for item in inputs]

#             assert isinstance(inputs, list), "inputs must be a list"
#             assert isinstance(inputs[0], dict), "inputs must be a list of dictionaries"

#         return func(f, inputs, *args, **kwargs)

#     return wrapper

def handle_inputs(f: Callable, inputs: Union[List[Dict[str, Any]], List[Any], pd.DataFrame]) -> List[Dict[str, Any]]:
    if f.__code__.co_argcount == 1:
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_dict("records")
        # logger.debug("Function has only 1 argument, converting list of dictionaries to list of values...")
        assert isinstance(inputs, list), "inputs must be a list"
        arg_name = f.__code__.co_varnames[0]
        inputs = [{arg_name: input_} for input_ in inputs]
    else:
        if isinstance(inputs, pd.DataFrame):
            # logger.debug("Converting DataFrame to list of dictionaries...")
            inputs = inputs.to_dict("records")
        elif isinstance(inputs[0], (list, tuple)):
            args_names = f.__code__.co_varnames[: f.__code__.co_argcount]
            inputs = [{argname: i for argname, i in zip(args_names, item)} for item in inputs]

        assert isinstance(inputs, list), "inputs must be a list"

    return inputs
    
    