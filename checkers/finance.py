"""
Finance task checker for tool-use tasks.
"""

from typing import Any, Dict, List


def check_finance_task(
    final_answer: str,
    gold_answer: Dict[str, Any],
    trace: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Check a finance-related task.
    
    Args:
        final_answer: Agent's final answer
        gold_answer: Gold answer with expected values
        trace: Full execution trace
        
    Returns:
        Dictionary with metrics
    """
    # Extract expected values from gold answer
    expected_answer = gold_answer.get("final_answer", "")
    required_tool_calls = gold_answer.get("required_tool_calls", [])
    
    # Check if answer matches
    answer_match = 1.0 if expected_answer.lower() in final_answer.lower() else 0.0
    
    # Check tool calls from trace
    tool_calls_made = []
    for event in trace:
        if event.get("event_type") == "TOOL_CALL":
            payload = event.get("payload", {})
            tool_name = payload.get("tool_name", "")
            tool_args = payload.get("arguments", {})
            tool_calls_made.append(f"{tool_name}({tool_args})")
    
    # Check if required tools were called
    tool_correctness = 0.0
    if required_tool_calls:
        matched_tools = sum(1 for req in required_tool_calls 
                          if any(req.lower() in call.lower() for call in tool_calls_made))
        tool_correctness = matched_tools / len(required_tool_calls)
    
    # Overall pass/fail
    pass_fail = 1.0 if (answer_match >= 0.8 and tool_correctness >= 0.8) else 0.0
    
    return {
        "pass_fail": pass_fail,
        "answer_match": answer_match,
        "tool_correctness": tool_correctness
    }
