"""
File: file.py
Author: Mathis Picasse
Description: useful functions to work with files.
"""

import logging
import os
from typing import Dict, List, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_output_dirs(
    root_dir: str,
    sub_dirs: List[str] = ['images', 'labels']
) -> Dict[str, str]:
    """
    Creates a root directory and a list of specified subdirectories within it.

    This function provides a robust way to prepare a directory structure,
    ensuring that all paths exist before they are used. It is idempotent,
    meaning it can be run multiple times without causing errors if the
    directories already exist.

    Args:
        root_dir (str): The path to the root directory where the subdirectories
            will be created.
        sub_dirs (List[str]): A list of names for the subdirectories to be
            created inside the root directory. Defaults to ['images', 'labels'].

    Returns:
        Dict[str, str]: A dictionary mapping each subdirectory name to its
            full, absolute path.

    Raises:
        OSError: If the function encounters a filesystem-related error, such
            as a lack of permissions to create a directory.

    """
    # Use a dictionary to store the resulting paths for clear, named access.
    created_paths: Dict[str, str] = {}

    logging.info(f"Setting up output directories in root: '{root_dir}'")

    # It's good practice to ensure the root directory itself exists.
    os.makedirs(root_dir, exist_ok=True)

    for dir_name in sub_dirs:
        # This loop creates each requested subdirectory.
        # This approach is more scalable than hardcoding directory names.
        full_path = os.path.join(root_dir, dir_name)
        try:
            os.makedirs(full_path, exist_ok=True)
            # Store the full path for the return value.
            created_paths[dir_name] = full_path
            logging.info(f"Ensured directory exists: '{full_path}'")
        except OSError as e:
            # This handles errors like permission denied, which `exist_ok=True`
            # does not suppress.
            logging.error(f"Could not create directory '{full_path}': {e}")
            raise  # Re-raise the exception to signal failure to the caller.

    return created_paths