from typing import Union, Optional
import os
from pathlib import Path

# Current working directory is path
DATA_ROOT: str = os.getcwd()


def set_dataset_path(path: Optional[Union[Path, str]] = None) -> None:
    """
    Set the dataset directory

    Parameters
    ----------
    path : Optional[Union[Path, str]]
        Direcotry path
        Defaults to None, which uses current working directory
    """
    global DATA_ROOT

    if path is None:
        path = os.getcwd()
    
    new_path = Path(path).absolute()
    if not new_path.exists():
        raise IOError(f"Path does not exist: {new_path}")

    DATA_ROOT = str(new_path)
    return
