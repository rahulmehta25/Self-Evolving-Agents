"""
Evaluator: Computes fitness metrics for completed agent runs.

Implements hard metrics (regex, json_schema, python_unit checkers) and
coordinates with JudgeEvaluator for soft metrics.
"""

import importlib
import json
import re
import time
from typing import Any, Dict, List, Optional

from jsonschema import validate, ValidationError


class Evaluator:
    """
    Evaluates agent runs and computes hard metrics.
    """
    
    def __init__(self, data_store):
        """
        Initialize the evaluator.
        
        Args:
            data_store: DataStore instance for retrieving traces
        """
        self.data_store = data_store
    
    def evaluate(self, run_id: str, task: Dict[str, Any], 
                run_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a completed run and compute all hard metrics.
        
        Args:
            run_id: ID of the run to evaluate
            task: Task specification
            run_result: Run result from AgentRunner
            
        Returns:
            Dictionary of metric_name -> metric_value
        """
        metrics = {}
        
        # 1. Budget/Status Metrics
        metrics["status_success"] = 1.0 if run_result.get("status") == "SUCCESS" else 0.0
        metrics["token_count"] = float(run_result.get("tokens_used", 0))
        metrics["tool_calls_count"] = float(run_result.get("tool_calls_used", 0))
        
        # Calculate latency if available
        run_record = self.data_store.get_run(run_id)
        if run_record and run_record.get("start_timestamp") and run_record.get("end_timestamp"):
            from datetime import datetime
            start = datetime.fromisoformat(run_record["start_timestamp"])
            end = datetime.fromisoformat(run_record["end_timestamp"])
            metrics["latency_seconds"] = (end - start).total_seconds()
        else:
            metrics["latency_seconds"] = 0.0
        
        # 2. Task-Specific Checker
        checker_type = task.get("checker_type")
        final_answer = run_result.get("final_response", "")
        
        if checker_type == "python_unit":
            checker_metrics = self._evaluate_python_unit(
                final_answer, task, run_id
            )
            metrics.update(checker_metrics)
        
        elif checker_type == "regex":
            checker_metrics = self._evaluate_regex(
                final_answer, task
            )
            metrics.update(checker_metrics)
        
        elif checker_type == "json_schema":
            checker_metrics = self._evaluate_json_schema(
                final_answer, task
            )
            metrics.update(checker_metrics)
        
        elif checker_type == "llm_judge_only":
            # Placeholder - will be handled by JudgeEvaluator
            metrics["pass_fail"] = 0.5  # Neutral placeholder
        
        else:
            # Unknown checker type
            metrics["pass_fail"] = 0.0
        
        return metrics
    
    def _evaluate_regex(self, final_answer: str, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate using regex pattern matching.
        
        Args:
            final_answer: Agent's final answer
            task: Task specification with checker_config
            
        Returns:
            Dictionary with pass_fail metric
        """
        checker_config = task.get("checker_config", {})
        pattern = checker_config.get("pattern", "")
        group = checker_config.get("group", 0)
        
        if not pattern:
            return {"pass_fail": 0.0}
        
        try:
            match = re.search(pattern, final_answer)
            if match:
                # Extract matched group if specified
                if group > 0 and match.groups():
                    matched_text = match.group(group)
                    return {"pass_fail": 1.0, "matched_text": matched_text}
                else:
                    return {"pass_fail": 1.0}
            else:
                return {"pass_fail": 0.0}
        except re.error as e:
            # Invalid regex pattern
            return {"pass_fail": 0.0, "error": str(e)}
    
    def _evaluate_json_schema(self, final_answer: str, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate using JSON schema validation.
        
        Args:
            final_answer: Agent's final answer (should be JSON)
            task: Task specification with checker_config containing schema
            
        Returns:
            Dictionary with pass_fail and validation details
        """
        checker_config = task.get("checker_config", {})
        schema = checker_config.get("schema", {})
        
        if not schema:
            return {"pass_fail": 0.0, "error": "No schema provided"}
        
        try:
            # Try to parse as JSON
            answer_json = json.loads(final_answer)
            
            # Validate against schema
            validate(instance=answer_json, schema=schema)
            return {"pass_fail": 1.0}
        
        except json.JSONDecodeError:
            return {"pass_fail": 0.0, "error": "Invalid JSON"}
        except ValidationError as e:
            return {"pass_fail": 0.0, "error": str(e)}
        except Exception as e:
            return {"pass_fail": 0.0, "error": str(e)}
    
    def _evaluate_python_unit(self, final_answer: str, task: Dict[str, Any],
                              run_id: str) -> Dict[str, float]:
        """
        Evaluate using a custom Python checker function.
        
        Args:
            final_answer: Agent's final answer
            task: Task specification with checker_config
            run_id: Run ID for retrieving trace
            
        Returns:
            Dictionary of metrics from the checker function
        """
        checker_config = task.get("checker_config", {})
        module_name = checker_config.get("module", "")
        function_name = checker_config.get("function", "")
        
        if not module_name or not function_name:
            return {"pass_fail": 0.0, "error": "Missing module or function name"}
        
        try:
            # Import the checker module
            checker_module = importlib.import_module(module_name)
            checker_func = getattr(checker_module, function_name)
            
            # Get the trace for the run
            trace_id = self.data_store.get_trace_id(run_id)
            trace = self.data_store.get_trace(trace_id) if trace_id else []
            
            # Get gold answer
            gold_answer = task.get("gold_answer", {})
            
            # Call the checker function
            result = checker_func(final_answer, gold_answer, trace)
            
            # Ensure result is a dictionary with float values
            if isinstance(result, dict):
                return {k: float(v) for k, v in result.items()}
            else:
                return {"pass_fail": float(result) if isinstance(result, (int, float)) else 0.0}
        
        except ImportError as e:
            return {"pass_fail": 0.0, "error": f"Failed to import module: {str(e)}"}
        except AttributeError as e:
            return {"pass_fail": 0.0, "error": f"Function not found: {str(e)}"}
        except Exception as e:
            return {"pass_fail": 0.0, "error": f"Checker error: {str(e)}"}
    
    def compute_weighted_fitness(self, metrics: Dict[str, float],
                                weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted fitness score from metrics.
        
        Args:
            metrics: Dictionary of metric_name -> metric_value
            weights: Optional dictionary of metric_name -> weight.
                    If None, uses default weights from the guide.
        
        Returns:
            Weighted fitness score
        """
        if weights is None:
            # Default weights from Section 5.6 of the guide
            weights = {
                "pass_fail": 0.5,
                "citation_fidelity": 0.3,
                "coherence_score": 0.1,
                "latency_seconds": -0.1  # Negative weight (penalty)
            }
        
        fitness = 0.0
        for metric_name, metric_value in metrics.items():
            weight = weights.get(metric_name, 0.0)
            fitness += weight * metric_value
        
        return fitness
