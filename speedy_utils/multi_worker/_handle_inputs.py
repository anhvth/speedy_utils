from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Union

import pandas as pd

# Example object
my_object = range(3)


def handle_inputs(f: Callable, inputs: Union[List[Dict[str, Any]], List[Any], pd.DataFrame]) -> List[Dict[str, Any]]:
    if isinstance(inputs, range | list | tuple):
        inputs = list(inputs)
                
    # Check if the object is iterable)
    if f.__code__.co_argcount == 1:
        if isinstance(inputs, pd.DataFrame):
            inputs = [r for _, r in inputs.iterrows()]
        assert isinstance(inputs, list), "inputs must be a list"
        arg_name = f.__code__.co_varnames[0]
        inputs = [{arg_name: input_} for input_ in inputs]
    else:
        raise NotImplementedError("Function has more than one argument, not implemented yet.")
        if isinstance(inputs, pd.DataFrame):
            # logger.debug("Converting DataFrame to list of dictionaries...")
            inputs = inputs.to_dict("records")
        elif isinstance(inputs[0], (list, tuple)):
            args_names = f.__code__.co_varnames[: f.__code__.co_argcount]
            inputs = [{argname: i for argname, i in zip(args_names, item)} for item in inputs]

        assert isinstance(inputs, list), "inputs must be a list"

    return inputs
