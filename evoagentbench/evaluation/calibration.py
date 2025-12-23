"""
Calibration utilities for LLM-as-Judge.

Implements calibration suite management and drift detection.
"""

from typing import Any, Dict, List, Optional


class CalibrationSuite:
    """
    Manages calibration suite for judge evaluation.
    """
    
    def __init__(self):
        """Initialize the calibration suite."""
        self.calibration_tasks: List[Dict[str, Any]] = []
        self.expert_scores: Dict[str, Dict[str, float]] = {}
    
    def add_calibration_task(self, task_id: str, expert_scores: Dict[str, float]):
        """
        Add a calibration task with expert scores.
        
        Args:
            task_id: Task ID
            expert_scores: Expert-provided scores for the task
        """
        self.expert_scores[task_id] = expert_scores
    
    def check_drift(self, judge_scores: Dict[str, Dict[str, float]],
                   threshold: float = 1.0) -> Dict[str, Any]:
        """
        Check for drift in judge scores compared to expert scores.
        
        Args:
            judge_scores: Judge scores by task_id
            threshold: Threshold for mean absolute error (default: 1.0)
            
        Returns:
            Drift detection report
        """
        errors = []
        
        for task_id, expert_score in self.expert_scores.items():
            if task_id in judge_scores:
                judge_score = judge_scores[task_id]
                
                # Calculate mean absolute error
                mae = 0.0
                count = 0
                for metric in expert_score:
                    if metric in judge_score:
                        mae += abs(expert_score[metric] - judge_score[metric])
                        count += 1
                
                if count > 0:
                    mae /= count
                    errors.append(mae)
        
        if not errors:
            return {
                "drift_detected": False,
                "mean_absolute_error": 0.0,
                "message": "No calibration data available"
            }
        
        mean_error = sum(errors) / len(errors)
        drift_detected = mean_error > threshold
        
        return {
            "drift_detected": drift_detected,
            "mean_absolute_error": mean_error,
            "threshold": threshold,
            "message": "Drift detected" if drift_detected else "No significant drift"
        }
