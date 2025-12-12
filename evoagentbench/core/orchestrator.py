"""
Orchestrator: Main entry point for EvoAgentBench.

Manages the overall workflow, including loading benchmarks, initializing genomes,
and coordinating evaluation and evolution cycles.
"""

import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .data_store import DataStore
from .evolution_engine import EvolutionEngine
from ..evaluation.evaluator import Evaluator
from ..evaluation.judge_evaluator import JudgeEvaluator
from ..runner.agent_runner import AgentRunner


class EvoBenchOrchestrator:
    """
    Main orchestrator for the EvoAgentBench system.
    """
    
    def __init__(self, db_path: str = "evoagentbench.db",
                 benchmark_path: str = "benchmarks/v1.0"):
        """
        Initialize the orchestrator.
        
        Args:
            db_path: Path to the database file
            benchmark_path: Path to the benchmark suite directory
        """
        self.data_store = DataStore(db_path)
        self.benchmark_path = benchmark_path
        self.evolution_engine = EvolutionEngine(self.data_store)
        self.agent_runner = AgentRunner(self.data_store)
        self.evaluator = Evaluator(self.data_store)
        self.judge_evaluator = JudgeEvaluator()
    
    def load_benchmark_suite(self) -> List[Dict[str, Any]]:
        """
        Load all tasks from the benchmark suite.
        
        Returns:
            List of task specifications
        """
        tasks = []
        benchmark_dir = Path(self.benchmark_path)
        
        if not benchmark_dir.exists():
            raise ValueError(f"Benchmark directory not found: {benchmark_path}")
        
        # Load all YAML and JSON files
        for task_file in benchmark_dir.rglob("*.yaml"):
            with open(task_file, 'r') as f:
                task = yaml.safe_load(f)
                tasks.append(task)
                self.data_store.save_task(task)
        
        for task_file in benchmark_dir.rglob("*.json"):
            with open(task_file, 'r') as f:
                task = json.load(f)
                tasks.append(task)
                self.data_store.save_task(task)
        
        return tasks
    
    def initialize_genomes(self, initial_genomes: List[Dict[str, Any]],
                          generation: int = 0) -> List[str]:
        """
        Initialize the first generation of genomes.
        
        Args:
            initial_genomes: List of initial genome configurations
            generation: Generation number (default: 0)
            
        Returns:
            List of genome IDs
        """
        genome_ids = []
        for genome in initial_genomes:
            genome["generation"] = generation
            if "genome_id" not in genome:
                genome["genome_id"] = str(uuid.uuid4())
            genome_id = self.data_store.save_genome(genome)
            genome_ids.append(genome_id)
        
        return genome_ids
    
    def evaluate_genome(self, genome_id: str, task: Dict[str, Any],
                       run_seed: int, generation_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a genome against a task.
        
        Args:
            genome_id: ID of the genome to evaluate
            task: Task specification
            run_seed: Random seed for deterministic execution
            generation_id: Optional generation ID
            
        Returns:
            Evaluation results
        """
        # Get genome
        genome = self.data_store.get_genome(genome_id)
        if not genome:
            raise ValueError(f"Genome {genome_id} not found")
        
        # Run agent
        run_result = self.agent_runner.run_task(genome, task, run_seed, generation_id)
        run_id = run_result["run_id"]
        
        # Evaluate hard metrics
        hard_metrics = self.evaluator.evaluate(run_id, task, run_result)
        
        # Evaluate soft metrics (judge)
        if run_result.get("final_response"):
            soft_metrics = self.judge_evaluator.evaluate(
                task, run_result["final_response"]
            )
        else:
            soft_metrics = {}
        
        # Combine metrics
        all_metrics = {**hard_metrics, **soft_metrics}
        
        # Compute weighted fitness
        weighted_fitness = self.evaluator.compute_weighted_fitness(all_metrics)
        all_metrics["weighted_fitness"] = weighted_fitness
        
        # Save metrics
        self.data_store.save_metrics(run_id, all_metrics, is_hard_metric=True)
        # Save soft metrics separately
        for metric_name, metric_value in soft_metrics.items():
            self.data_store.save_metric(run_id, metric_name, metric_value, is_hard_metric=False)
        self.data_store.save_metric(run_id, "weighted_fitness", weighted_fitness, is_hard_metric=True)
        
        return {
            "run_id": run_id,
            "metrics": all_metrics,
            "status": run_result.get("status")
        }
    
    def evaluate_generation(self, generation: int,
                           tasks: List[Dict[str, Any]],
                           holdout_fraction: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate all genomes in a generation against all tasks.
        
        Args:
            generation: Generation number
            tasks: List of tasks to evaluate against
            holdout_fraction: Fraction of tasks to use as holdout (excluded from fitness)
            
        Returns:
            Evaluation summary
        """
        # Get genomes in generation
        genomes = self.data_store.get_genomes_by_generation(generation)
        
        if not genomes:
            raise ValueError(f"No genomes found for generation {generation}")
        
        # Split tasks into main and holdout
        random.shuffle(tasks)
        split_idx = int(len(tasks) * (1 - holdout_fraction))
        main_tasks = tasks[:split_idx]
        holdout_tasks = tasks[split_idx:]
        
        # Record generation start
        start_time = datetime.utcnow()
        self.data_store.save_generation(
            generation, start_time, population_size=len(genomes)
        )
        
        # Evaluate each genome against each task
        all_results = []
        for genome in genomes:
            genome_id = genome["genome_id"]
            for task in main_tasks:
                run_seed = random.randint(0, 2**31 - 1)
                result = self.evaluate_genome(genome_id, task, run_seed, generation)
                all_results.append(result)
        
        # Calculate average fitness
        fitness_scores = self.data_store.get_fitness_by_generation(
            generation, "weighted_fitness"
        )
        avg_fitness = sum(f for _, f in fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        
        # Get best genome
        best_genome_id = self.evolution_engine.get_best_genome(generation)
        
        # Update generation record
        end_time = datetime.utcnow()
        self.data_store.save_generation(
            generation, start_time, best_genome_id, avg_fitness, len(genomes)
        )
        
        return {
            "generation": generation,
            "population_size": len(genomes),
            "tasks_evaluated": len(main_tasks),
            "holdout_tasks": len(holdout_tasks),
            "avg_fitness": avg_fitness,
            "best_genome_id": best_genome_id,
            "results": all_results
        }
    
    def run_evolution(self, num_generations: int,
                     initial_genomes: List[Dict[str, Any]],
                     population_size: int = 20,
                     holdout_fraction: float = 0.1) -> Dict[str, Any]:
        """
        Run the full evolutionary loop.
        
        Args:
            num_generations: Number of generations to evolve
            initial_genomes: Initial genome configurations
            population_size: Population size per generation
            holdout_fraction: Fraction of tasks for holdout set
            
        Returns:
            Evolution summary
        """
        # Load benchmark suite
        tasks = self.load_benchmark_suite()
        
        # Initialize first generation
        genome_ids = self.initialize_genomes(initial_genomes, generation=0)
        
        evolution_history = []
        
        for generation in range(num_generations):
            print(f"Generation {generation}: Evaluating {len(genome_ids)} genomes...")
            
            # Evaluate generation
            eval_results = self.evaluate_generation(
                generation, tasks, holdout_fraction
            )
            evolution_history.append(eval_results)
            
            print(f"  Average fitness: {eval_results['avg_fitness']:.4f}")
            print(f"  Best genome: {eval_results['best_genome_id']}")
            
            # Evolve next generation (except for last generation)
            if generation < num_generations - 1:
                print(f"  Evolving generation {generation + 1}...")
                genome_ids = self.evolution_engine.evolve_generation(
                    generation, population_size
                )
                print(f"  Created {len(genome_ids)} new genomes")
        
        return {
            "num_generations": num_generations,
            "final_generation": num_generations - 1,
            "history": evolution_history
        }
