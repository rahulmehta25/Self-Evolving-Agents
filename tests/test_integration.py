"""
Integration tests for EvoAgentBench evolution loop.
"""

import os
import tempfile
import unittest

from evoagentbench.core.orchestrator import EvoBenchOrchestrator


class TestEvolutionIntegration(unittest.TestCase):
    """Integration tests for the full evolution loop."""

    def setUp(self):
        """Set up test fixtures with temp database and mock LLM."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.temp_benchmark = tempfile.mkdtemp()
        self._create_test_tasks()
        os.environ["EVOAGENTBENCH_LLM"] = "mock"
        os.environ["EVOAGENTBENCH_JUDGE_LLM"] = "mock"

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_db.name)
        import shutil
        shutil.rmtree(self.temp_benchmark, ignore_errors=True)
        os.environ.pop("EVOAGENTBENCH_LLM", None)
        os.environ.pop("EVOAGENTBENCH_JUDGE_LLM", None)

    def _create_test_tasks(self):
        """Create minimal test tasks in the temp benchmark directory."""
        import json
        task1 = {
            "task_id": "test_task_1",
            "version": 1,
            "category": ["test"],
            "difficulty": "easy",
            "checker_type": "regex",
            "prompt_template": "What is 2 + 2?",
            "gold_answer": {"pattern": "4|four"},
            "checker_config": {"pattern": "4|four"},
            "budget": {"max_tokens": 500, "max_tool_calls": 3, "max_time_seconds": 10}
        }
        task2 = {
            "task_id": "test_task_2",
            "version": 1,
            "category": ["test"],
            "difficulty": "easy",
            "checker_type": "regex",
            "prompt_template": "Say hello",
            "gold_answer": {"pattern": "hello|hi"},
            "checker_config": {"pattern": "hello|hi"},
            "budget": {"max_tokens": 500, "max_tool_calls": 3, "max_time_seconds": 10}
        }
        with open(os.path.join(self.temp_benchmark, "task1.json"), 'w') as f:
            json.dump(task1, f)
        with open(os.path.join(self.temp_benchmark, "task2.json"), 'w') as f:
            json.dump(task2, f)

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        orchestrator = EvoBenchOrchestrator(
            db_path=self.temp_db.name,
            benchmark_path=self.temp_benchmark,
            llm_provider="mock"
        )
        self.assertIsNotNone(orchestrator.data_store)
        self.assertIsNotNone(orchestrator.agent_runner)
        self.assertIsNotNone(orchestrator.evaluator)

    def test_load_benchmark_suite(self):
        """Test loading benchmark tasks."""
        orchestrator = EvoBenchOrchestrator(
            db_path=self.temp_db.name,
            benchmark_path=self.temp_benchmark,
            llm_provider="mock"
        )
        tasks = orchestrator.load_benchmark_suite()
        self.assertEqual(len(tasks), 2)

    def test_initialize_genomes(self):
        """Test genome initialization."""
        orchestrator = EvoBenchOrchestrator(
            db_path=self.temp_db.name,
            benchmark_path=self.temp_benchmark,
            llm_provider="mock"
        )
        genomes = [
            {"system_prompt": "You are a helpful assistant.", "llm_config": {"model_name": "mock", "temperature": 0.0}},
            {"system_prompt": "Think step by step.", "llm_config": {"model_name": "mock", "temperature": 0.5}}
        ]
        genome_ids = orchestrator.initialize_genomes(genomes, generation=0)
        self.assertEqual(len(genome_ids), 2)

    def test_evaluate_single_genome(self):
        """Test evaluating a single genome against a task."""
        orchestrator = EvoBenchOrchestrator(
            db_path=self.temp_db.name,
            benchmark_path=self.temp_benchmark,
            llm_provider="mock"
        )
        tasks = orchestrator.load_benchmark_suite()
        genomes = [{"system_prompt": "Test", "llm_config": {"model_name": "mock", "temperature": 0.0}}]
        genome_ids = orchestrator.initialize_genomes(genomes)
        result = orchestrator.evaluate_genome(genome_ids[0], tasks[0], run_seed=42)
        self.assertIn("run_id", result)
        self.assertIn("metrics", result)
        self.assertIn("weighted_fitness", result["metrics"])

    def test_evaluate_generation(self):
        """Test evaluating a full generation."""
        orchestrator = EvoBenchOrchestrator(
            db_path=self.temp_db.name,
            benchmark_path=self.temp_benchmark,
            llm_provider="mock"
        )
        tasks = orchestrator.load_benchmark_suite()
        genomes = [
            {"system_prompt": "Genome A", "llm_config": {"model_name": "mock", "temperature": 0.0}},
            {"system_prompt": "Genome B", "llm_config": {"model_name": "mock", "temperature": 0.5}}
        ]
        orchestrator.initialize_genomes(genomes, generation=0)
        result = orchestrator.evaluate_generation(0, tasks, holdout_fraction=0.0)
        self.assertEqual(result["generation"], 0)
        self.assertEqual(result["population_size"], 2)
        self.assertIn("avg_fitness", result)

    def test_mini_evolution_loop(self):
        """Test a minimal 2-generation evolution loop."""
        orchestrator = EvoBenchOrchestrator(
            db_path=self.temp_db.name,
            benchmark_path=self.temp_benchmark,
            llm_provider="mock"
        )
        tasks = orchestrator.load_benchmark_suite()
        initial_genomes = [
            {"system_prompt": "Be concise.", "llm_config": {"model_name": "mock", "temperature": 0.0}},
            {"system_prompt": "Be verbose.", "llm_config": {"model_name": "mock", "temperature": 0.7}}
        ]
        orchestrator.initialize_genomes(initial_genomes, generation=0)

        # Gen 0 evaluation
        gen0_result = orchestrator.evaluate_generation(0, tasks, holdout_fraction=0.0)
        self.assertIsNotNone(gen0_result["best_genome_id"])

        # Evolve to gen 1
        new_genome_ids = orchestrator.evolution_engine.evolve_generation(0, population_size=2)
        self.assertGreater(len(new_genome_ids), 0)

        # Evaluate gen 1
        gen1_result = orchestrator.evaluate_generation(1, tasks, holdout_fraction=0.0)
        self.assertEqual(gen1_result["generation"], 1)


if __name__ == "__main__":
    unittest.main()
