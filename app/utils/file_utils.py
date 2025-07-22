"""
File utility functions.
Handles common file operations and utilities.
"""

import os
import tempfile
from typing import Optional

def create_temp_directory(prefix: str = "sheetsloader_") -> str:
    """
    Create a temporary directory.
    
    Args:
        prefix (str): Prefix for the temporary directory name
        
    Returns:
        str: Path to the created temporary directory
    """
    return tempfile.mkdtemp(prefix=prefix)

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File extension (including the dot)
    """
    return os.path.splitext(file_path)[1]

def is_valid_file_path(file_path: str) -> bool:
    """
    Check if a file path is valid.
    
    Args:
        file_path (str): Path to check
        
    Returns:
        bool: True if the path is valid
    """
    try:
        # Check if the path is absolute or can be made absolute
        os.path.abspath(file_path)
        return True
    except Exception:
        return False 