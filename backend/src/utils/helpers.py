import os
import hashlib
import json
from typing import Any, Optional
from loguru import logger

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""

def ensure_dir_exists(directory: str) -> None:
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)

def save_json(data: Any, file_path: str) -> bool:
    """Save data to JSON file."""
    try:
        ensure_dir_exists(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False

def load_json(file_path: str) -> Optional[Any]:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info(f"JSON file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename[:255]  # Limit to max filename length 