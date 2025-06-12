# jupyter notebook utilities
import os
import pathlib


def change_dir(target_directory: str = 'POLY') -> None:
    """Change directory to the first occurrence of x in the current path."""
    cur_dir = pathlib.Path('./')
    target_dir = str(cur_dir.absolute()).split(target_directory)[0] + target_directory
    os.chdir(target_dir)
    print(f'Current dir: {target_dir}')