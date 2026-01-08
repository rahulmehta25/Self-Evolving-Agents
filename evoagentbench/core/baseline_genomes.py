"""
Baseline genome configurations for comparison.

Implements the three baseline agents from the guide:
1. Zero-Shot Baseline
2. ReAct Baseline
3. Previous Best Baseline
"""

import uuid
from typing import Dict, Any


def create_zero_shot_baseline() -> Dict[str, Any]:
    """
    Create a zero-shot baseline genome.
    
    Returns:
        Zero-shot baseline genome configuration
    """
    return {
        "genome_id": f"baseline_zero_shot_{uuid.uuid4()}",
        "parent_id": None,
        "generation": 0,
        "system_prompt": "You are a helpful AI assistant.",
        "llm_config": {
            "model_name": "gpt-4",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1000
        },
        "tools": [],
        "planner_type": "none",
        "metadata": {
            "baseline_type": "zero_shot",
            "description": "Simplest agent with no tools, memory, or planning"
        }
    }


def create_react_baseline() -> Dict[str, Any]:
    """
    Create a ReAct baseline genome.
    
    Returns:
        ReAct baseline genome configuration
    """
    return {
        "genome_id": f"baseline_react_{uuid.uuid4()}",
        "parent_id": None,
        "generation": 0,
        "system_prompt": """You are a helpful AI assistant that can use tools to answer questions.
Think step by step:
1. First, understand the question
2. Decide if you need to use a tool
3. If yes, call the tool and observe the result
4. Use the result to formulate your final answer""",
        "llm_config": {
            "model_name": "gpt-4",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2000
        },
        "tools": [],
        "planner_type": "react",
        "metadata": {
            "baseline_type": "react",
            "description": "Standard ReAct architecture with step-by-step reasoning"
        }
    }


def create_previous_best_baseline(previous_genome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a baseline from a previous best genome.
    
    Args:
        previous_genome: Previous best genome configuration
        
    Returns:
        Previous best baseline genome configuration
    """
    import copy
    baseline = copy.deepcopy(previous_genome)
    baseline["genome_id"] = f"baseline_previous_best_{uuid.uuid4()}"
    baseline["parent_id"] = previous_genome.get("genome_id")
    baseline["generation"] = 0
    baseline["metadata"] = baseline.get("metadata", {})
    baseline["metadata"]["baseline_type"] = "previous_best"
    baseline["metadata"]["previous_genome_id"] = previous_genome.get("genome_id")
    
    return baseline
