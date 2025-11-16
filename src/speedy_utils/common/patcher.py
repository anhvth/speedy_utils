# utils/patching.py
from ..__imports import *


def patch_method(
    cls: Annotated[type, 'Class containing the method'],
    method_name: Annotated[str, 'Name of the method to patch'],
    replacements: Annotated[
        dict[str | re.Pattern, str],
        'Mapping of {old_substring_or_regex: new_string} replacements',
    ],
    tag: Annotated[str, 'Optional logging tag'] = '',
) -> bool:
    """
    Generic patcher for replacing substrings or regex matches in a method's source code.

    Args:
        cls: Target class
        method_name: Method name to patch
        replacements: {pattern: replacement}. Patterns may be plain strings or regex patterns.
        tag: Optional string shown in logs

    Returns:
        bool: True if successfully patched, False otherwise
    """

    try:
        method = getattr(cls, method_name)
    except AttributeError:
        print(
            f'[patcher{":" + tag if tag else ""}] No method {method_name} in {cls.__name__}'
        )
        return False

    try:
        src = inspect.getsource(method)
    except (OSError, TypeError):
        print(
            f'[patcher{":" + tag if tag else ""}] Could not get source for {cls.__name__}.{method_name}'
        )
        return False

    new_src = src
    did_patch = False

    for old, new in replacements.items():
        if isinstance(old, re.Pattern):
            if old.search(new_src):
                new_src = old.sub(new, new_src)
                did_patch = True
        elif isinstance(old, str):
            if old in new_src:
                new_src = new_src.replace(old, new)
                did_patch = True
        else:
            raise TypeError('Replacement keys must be str or re.Pattern')

    if not did_patch:
        print(
            f'[patcher{":" + tag if tag else ""}] No matching patterns found in {cls.__name__}.{method_name}'
        )
        return False

    # Recompile the patched function
    code_obj = compile(new_src, filename=f'<patched_{method_name}>', mode='exec')
    ns = {}
    exec(code_obj, cls.__dict__, ns)  # type: ignore

    # Attach patched method back
    setattr(cls, method_name, types.MethodType(ns[method_name], None, cls))  # type: ignore
    print(f'[patcher{":" + tag if tag else ""}] Patched {cls.__name__}.{method_name}')
    return True
