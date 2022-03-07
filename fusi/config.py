from typing import Union
import os
from pathlib import Path

# Current working directory is path
DATA_ROOT: str = os.getcwd()


def set_dataset_path(path: Union[Path, str]) -> None:
    """
    Set the dataset directory
    """
    global DATA_ROOT
    
    new_path = Path(path).absolute()
    if not new_path.exists():
        raise IOError(f"Path does not exist: {new_path}")

    DATA_ROOT = str(new_path)
    return
