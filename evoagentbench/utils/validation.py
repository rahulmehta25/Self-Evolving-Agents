"""
Validation utilities for schemas and configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import validate, ValidationError


def validate_task_spec(task: Dict[str, Any], schema_path: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    Validate a task specification against the schema.
    
    Args:
        task: Task specification dictionary
        schema_path: Optional path to schema file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if schema_path is None:
        schema_path = "evoagentbench/schemas/task_spec_v1.json"
    
    schema_file = Path(schema_path)
    if not schema_file.exists():
        return False, f"Schema file not found: {schema_path}"
    
    try:
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        validate(instance=task, schema=schema)
        return True, None
    
    except ValidationError as e:
        return False, f"Validation error: {e.message}"
    except Exception as e:
        return False, f"Error loading schema: {str(e)}"


def validate_genome_spec(genome: Dict[str, Any], schema_path: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    Validate a genome specification against the schema.
    
    Args:
        genome: Genome specification dictionary
        schema_path: Optional path to schema file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if schema_path is None:
        schema_path = "evoagentbench/schemas/genome_spec_v1.json"
    
    schema_file = Path(schema_path)
    if not schema_file.exists():
        return False, f"Schema file not found: {schema_path}"
    
    try:
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        validate(instance=genome, schema=schema)
        return True, None
    
    except ValidationError as e:
        return False, f"Validation error: {e.message}"
    except Exception as e:
        return False, f"Error loading schema: {str(e)}"
