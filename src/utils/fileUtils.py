import os, time, shutil
from typing import Generator

def traverse_file(path: str, extname: str | list[str] | tuple[str, ...]) -> Generator[tuple[str, str], None, None]:
    """
    Traverse all files with specified extension name in the specified path.

    Args:
        path (str): The path to be traversed.
        extname (str, list or tuple): The extension name of the files to be traversed.
    
    Returns:
        Generator[tuple[str, str], None, None]: A generator that yields the tuple of the root path and the file name.
    """

    extnames: tuple[str, ...]
    if isinstance(extname, str):
        extnames = (extname,)
    elif isinstance(extname, list):
        extnames = tuple(extname)
    elif isinstance(extname, tuple):
        extnames = extname
    else:
        raise TypeError("The arg `extname` must be str, list or tuple")

    w = os.walk(path)

    for root, _, files in w:
        for f in files:
            if f.endswith(extnames):
                yield root, f

def backup_file(path: str):
    """
    Backup the file with the current time appended to the file name.
    
    Args:
        path (str): The path of the file to be backed up.
    """

    if os.path.exists(path):
        dot_pos = path.rfind('.')
        cur_time = time.strftime('_%Y%m%d_%H%M%S')
        backup_path = path[:dot_pos] + cur_time + path[dot_pos:]
        shutil.copy(path, backup_path)
    else:
        raise FileNotFoundError(f"File not found: {path}")
