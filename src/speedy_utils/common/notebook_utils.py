# jupyter notebook utilities
import os
import pathlib
from typing import str as StrType


def change_dir(x: StrType = 'POLY') -> None:
    """Change directory to the first occurrence of x in the current path."""
    cur_dir = pathlib.Path('./')
    target_dir = str(cur_dir.absolute()).split(x)[0] + x
    os.chdir(target_dir)
    print(f'Current dir: {target_dir}')