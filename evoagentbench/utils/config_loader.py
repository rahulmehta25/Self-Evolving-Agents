"""
Configuration loader for genomes and tasks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_genome_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load a genome configuration from a file.
    
    Args:
        file_path: Path to genome file (JSON or YAML)
        
    Returns:
        Genome configuration dictionary
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Genome file not found: {file_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def load_genomes_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Load all genome configurations from a directory.
    
    Args:
        directory: Directory containing genome files
        
    Returns:
        List of genome configurations
    """
    genomes = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for file_path in dir_path.glob("*.json"):
        genomes.append(load_genome_from_file(str(file_path)))
    
    for file_path in dir_path.glob("*.yaml"):
        genomes.append(load_genome_from_file(str(file_path)))
    
    for file_path in dir_path.glob("*.yml"):
        genomes.append(load_genome_from_file(str(file_path)))
    
    return genomes
