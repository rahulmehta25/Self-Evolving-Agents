"""
Hash utilities for deterministic hashing of inputs/outputs.
"""

import hashlib
import json
from typing import Any


def compute_hash(data: Any) -> str:
    """
    Compute SHA256 hash of JSON-serializable data.
    
    Args:
        data: Data to hash (dict, list, str, etc.)
        
    Returns:
        Hexadecimal hash string
    """
    if data is None:
        return ""
    
    if isinstance(data, (dict, list)):
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    return hashlib.sha256(str(data).encode()).hexdigest()


def verify_hash(data: Any, expected_hash: str) -> bool:
    """
    Verify that data matches expected hash.
    
    Args:
        data: Data to verify
        expected_hash: Expected hash value
        
    Returns:
        True if hash matches, False otherwise
    """
    computed_hash = compute_hash(data)
    return computed_hash == expected_hash
