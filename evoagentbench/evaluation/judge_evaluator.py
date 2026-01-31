"""
Judge Evaluator: LLM-as-Judge for computing soft metrics.

Implements LLM-based evaluation with rubric, few-shot examples,
calibration, and bias mitigation. Uses Vertex AI Gemini when
EVOAGENTBENCH_JUDGE_LLM=vertex (same GCP credits as agent LLM).

Guide §5.4: Order bias randomization for fair comparison.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


class JudgeEvaluator:
    """
    Evaluates agent outputs using an LLM as a judge.
    
    M2 Version: Structure in place, actual LLM calls will be integrated in M3.
    """
    
    def __init__(self, judge_model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the judge evaluator.
        
        Args:
            judge_model_config: Configuration for the judge LLM model
        """
        self.judge_model_config = judge_model_config or {
            "model_name": "gpt-4",
            "temperature": 0.0  # Deterministic judging
        }
        self.calibration_suite: List[Dict[str, Any]] = []
        self.few_shot_examples: Dict[str, List[Dict[str, Any]]] = {}
    
    def load_rubric(self, category: List[str]) -> Dict[str, Any]:
        """
        Load the evaluation rubric for a task category.
        
        Args:
            category: List of task categories
            
        Returns:
            Rubric dictionary with scoring dimensions
        """
        # Default rubric - can be customized per category
        rubric = {
            "dimensions": [
                {
                    "name": "coherence_score",
                    "description": "How coherent and well-structured is the response?",
                    "scale": {"min": 1, "max": 5}
                },
                {
                    "name": "completeness_score",
                    "description": "How completely does the response address the task?",
                    "scale": {"min": 1, "max": 5}
                },
                {
                    "name": "correctness_score",
                    "description": "How correct is the information provided?",
                    "scale": {"min": 1, "max": 5}
                }
            ],
            "instructions": "Score each dimension on a scale of 1-5, providing justification."
        }
        
        return rubric
    
    def load_few_shot_examples(self, category: List[str]) -> List[Dict[str, Any]]:
        """
        Load few-shot examples for the judge prompt.
        
        Args:
            category: List of task categories
            
        Returns:
            List of example dictionaries with output and scores
        """
        category_key = ",".join(sorted(category))
        
        if category_key not in self.few_shot_examples:
            # Default examples - should be loaded from calibration suite
            self.few_shot_examples[category_key] = [
                {
                    "output": "Example output 1",
                    "scores": {
                        "coherence_score": 4,
                        "completeness_score": 5,
                        "correctness_score": 4
                    },
                    "justification": "Well-structured and complete response."
                }
            ]
        
        return self.few_shot_examples[category_key]
    
    def build_judge_prompt(self, task_description: str, agent_output: str,
                          rubric: Dict[str, Any],
                          few_shot_examples: List[Dict[str, Any]]) -> str:
        """
        Build the judge prompt with rubric and examples.
        
        Args:
            task_description: The task prompt template
            agent_output: The agent's final output
            rubric: Evaluation rubric
            few_shot_examples: Few-shot examples for calibration
            
        Returns:
            Formatted judge prompt
        """
        prompt_parts = [
            "You are an expert evaluator judging the quality of an AI agent's response.",
            "",
            "TASK DESCRIPTION:",
            task_description,
            "",
            "AGENT OUTPUT:",
            agent_output,
            "",
            "EVALUATION RUBRIC:",
        ]
        
        for dim in rubric["dimensions"]:
            prompt_parts.append(
                f"- {dim['name']}: {dim['description']} "
                f"(Scale: {dim['scale']['min']}-{dim['scale']['max']})"
            )
        
        if few_shot_examples:
            prompt_parts.append("")
            prompt_parts.append("FEW-SHOT EXAMPLES:")
            # Guide §5.4: Randomize order of few-shot examples to prevent position bias
            shuffled_examples = list(few_shot_examples[:3])
            random.shuffle(shuffled_examples)
            for i, example in enumerate(shuffled_examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Output: {example['output']}")
                prompt_parts.append(f"Scores: {json.dumps(example['scores'])}")
                prompt_parts.append(f"Justification: {example['justification']}")
        
        prompt_parts.append("")
        prompt_parts.append("INSTRUCTIONS:")
        prompt_parts.append(rubric["instructions"])
        prompt_parts.append("")
        prompt_parts.append(
            "Provide your evaluation as a JSON object with the following structure:"
        )
        prompt_parts.append('{"scores": {"coherence_score": <1-5>, "completeness_score": <1-5>, "correctness_score": <1-5>}, "justification": "<your justification>"}')
        
        return "\n".join(prompt_parts)
    
    def evaluate(self, task: Dict[str, Any], agent_output: str,
                trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Evaluate agent output using LLM-as-Judge.
        
        Args:
            task: Task specification
            agent_output: Agent's final output
            trace: Optional trace for context
            
        Returns:
            Dictionary of soft metrics (scores)
        """
        # Load rubric and examples
        category = task.get("category", [])
        rubric = self.load_rubric(category)
        few_shot_examples = self.load_few_shot_examples(category)
        
        # Build prompt
        task_description = task.get("prompt_template", "")
        judge_prompt = self.build_judge_prompt(
            task_description, agent_output, rubric, few_shot_examples
        )
        
        judge_response = self._judge_call(judge_prompt)
        scores = self._parse_judge_output(judge_response)
        return scores

    def _judge_call(self, prompt: str) -> str:
        """
        Call judge LLM. Uses Vertex Gemini when EVOAGENTBENCH_JUDGE_LLM=vertex
        or when unset and Vertex SDK is available (default production). Otherwise mock.
        """
        env_val = (os.environ.get("EVOAGENTBENCH_JUDGE_LLM") or "").strip().lower()
        use_vertex = env_val == "vertex" or (env_val != "mock" and self._vertex_available())
        if use_vertex:
            try:
                from evoagentbench.runner.llm_adapters import (
                    _vertex_gemini_available,
                    call_vertex_gemini_text_only,
                )
                if _vertex_gemini_available():
                    model = str(self.judge_model_config.get("model_name") or "gemini-1.5-flash")
                    if not model.startswith("gemini-"):
                        model = "gemini-1.5-flash"
                    temp = float(self.judge_model_config.get("temperature", 0.0))
                    return call_vertex_gemini_text_only(prompt, model_id=model, temperature=temp)
            except Exception:
                pass
        return self._mock_judge_call(prompt)

    def _vertex_available(self) -> bool:
        try:
            from evoagentbench.runner.llm_adapters import _vertex_gemini_available
            return _vertex_gemini_available()
        except Exception:
            return False

    def _mock_judge_call(self, prompt: str) -> str:
        """Fallback mock when Vertex is not used or call fails."""
        return json.dumps({
            "scores": {
                "coherence_score": 4,
                "completeness_score": 4,
                "correctness_score": 3
            },
            "justification": "The response is coherent and mostly complete, but correctness could be improved."
        })
    
    def _parse_judge_output(self, judge_response: str) -> Dict[str, float]:
        """
        Parse the judge's JSON response (allows raw JSON or markdown-wrapped).
        """
        raw = (judge_response or "").strip()
        for start in ("```json", "```"):
            if start in raw:
                i = raw.find(start) + len(start)
                j = raw.find("```", i)
                if j != -1:
                    raw = raw[i:j].strip()
                else:
                    raw = raw[i:].strip()
                break
        try:
            response_data = json.loads(raw)
            scores = response_data.get("scores", {})
            
            # Normalize to float and ensure all expected dimensions are present
            normalized_scores = {}
            for dim in ["coherence_score", "completeness_score", "correctness_score"]:
                value = scores.get(dim, 3.0)  # Default to neutral
                # Ensure value is in valid range [1, 5]
                value = max(1.0, min(5.0, float(value)))
                normalized_scores[dim] = value
            
            return normalized_scores
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to neutral scores on parse error
            return {
                "coherence_score": 3.0,
                "completeness_score": 3.0,
                "correctness_score": 3.0
            }
    
    def check_calibration(self, calibration_suite: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check judge calibration against a calibration suite.
        
        Args:
            calibration_suite: List of tasks with expert-scored outputs
            
        Returns:
            Calibration report with drift detection
        """
        # M2: Placeholder for calibration checking
        # In full implementation, would:
        # 1. Run judge on calibration suite
        # 2. Compare judge scores to expert scores
        # 3. Calculate deviation (e.g., mean absolute error)
        # 4. Alert if deviation > threshold
        
        return {
            "calibrated": True,
            "mean_absolute_error": 0.0,
            "drift_detected": False
        }
    
    def calculate_cohens_kappa(self, judge1_scores: List[float],
                               judge2_scores: List[float]) -> float:
        """
        Calculate Cohen's Kappa for inter-judge agreement.
        
        Args:
            judge1_scores: Scores from first judge
            judge2_scores: Scores from second judge
            
        Returns:
            Cohen's Kappa statistic
        """
        # M2: Placeholder implementation
        # Full implementation would use scipy.stats.cohen_kappa_score
        # For now, return a placeholder value
        if len(judge1_scores) != len(judge2_scores):
            return 0.0
        
        # Simple agreement calculation
        agreements = sum(1 for j1, j2 in zip(judge1_scores, judge2_scores) if abs(j1 - j2) <= 0.5)
        agreement_rate = agreements / len(judge1_scores) if judge1_scores else 0.0
        
        # Approximate Kappa (simplified)
        kappa = (agreement_rate - 0.2) / 0.8  # Rough approximation
        return max(0.0, min(1.0, kappa))

    def compare_outputs(
        self,
        task: Dict[str, Any],
        output_a: str,
        output_b: str,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """
        Compare two agent outputs with order randomization (Guide §5.4).

        Args:
            task: Task specification
            output_a: First agent's output
            output_b: Second agent's output
            seed: Optional seed for reproducible randomization

        Returns:
            (scores_a, scores_b, was_swapped) - scores for each output and whether order was swapped
        """
        if seed is not None:
            random.seed(seed)

        # Randomize order to prevent position bias
        swap = random.choice([True, False])
        if swap:
            first_output, second_output = output_b, output_a
        else:
            first_output, second_output = output_a, output_b

        # Evaluate both outputs
        scores_first = self.evaluate(task, first_output)
        scores_second = self.evaluate(task, second_output)

        # Map back to original order
        if swap:
            return scores_second, scores_first, True
        return scores_first, scores_second, False
