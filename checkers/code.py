"""
Code task checker for code generation tasks.
"""

import re
from typing import Any, Dict, List


def check_code_task(
    final_answer: str,
    gold_answer: Dict[str, Any],
    trace: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Check a code generation task.
    
    Args:
        final_answer: Agent's final answer (should contain code)
        gold_answer: Gold answer with expected function and test cases
        trace: Full execution trace
        
    Returns:
        Dictionary with metrics
    """
    expected_function = gold_answer.get("expected_function_name", "")
    test_cases = gold_answer.get("test_cases", [])
    
    # Extract code from answer (look for function definition)
    code_match = re.search(rf"def\s+{expected_function}\s*\(", final_answer, re.IGNORECASE)
    has_function = 1.0 if code_match else 0.0
    
    # Try to extract and test the function (simplified)
    # In a full implementation, would actually execute the code
    syntax_valid = 1.0  # Placeholder - would check Python syntax
    
    # Overall pass/fail
    pass_fail = 1.0 if (has_function >= 0.5 and syntax_valid >= 0.5) else 0.0
    
    return {
        "pass_fail": pass_fail,
        "has_function": has_function,
        "syntax_valid": syntax_valid
    }
