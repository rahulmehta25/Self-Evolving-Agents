"""
Generate statistical reports for evolutionary runs.

Implements reporting with bootstrap confidence intervals and significance testing.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evoagentbench.core.data_store import DataStore
from evoagentbench.evaluation.stats_module import (
    calculate_aggregate_metrics,
    calculate_bootstrap_ci,
    permutation_test,
    win_tie_loss_analysis
)


def generate_generation_report(data_store: DataStore, generation: int) -> Dict[str, Any]:
    """
    Generate a report for a specific generation.
    
    Args:
        data_store: DataStore instance
        generation: Generation number
        
    Returns:
        Report dictionary
    """
    # Get all genomes in generation
    genomes = data_store.get_genomes_by_generation(generation)
    
    # Get fitness scores
    fitness_scores = data_store.get_fitness_by_generation(generation, "weighted_fitness")
    
    # Get all runs for this generation
    cursor = data_store.conn.cursor()
    cursor.execute("""
        SELECT r.run_id, r.genome_id, r.task_id, r.status
        FROM runs r
        JOIN genomes g ON r.genome_id = g.genome_id
        WHERE g.generation = ?
    """, (generation,))
    
    runs = [dict(row) for row in cursor.fetchall()]
    
    # Aggregate metrics by task category
    task_performance = {}
    for run in runs:
        run_id = run["run_id"]
        task_id = run["task_id"]
        metrics = data_store.get_metrics(run_id)
        
        # Get task category
        cursor.execute("SELECT category FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        category = row[0] if row else "unknown"
        
        if category not in task_performance:
            task_performance[category] = []
        
        task_performance[category].append(metrics)
    
    # Calculate aggregate statistics
    report = {
        "generation": generation,
        "population_size": len(genomes),
        "fitness_statistics": {},
        "task_performance": {}
    }
    
    # Fitness statistics
    if fitness_scores:
        fitness_values = [f for _, f in fitness_scores]
        ci_lower, ci_upper, mean = calculate_bootstrap_ci(fitness_values)
        report["fitness_statistics"] = {
            "mean": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "min": min(fitness_values),
            "max": max(fitness_values)
        }
    
    # Task performance by category
    for category, metrics_list in task_performance.items():
        if metrics_list:
            pass_fail_values = [m.get("pass_fail", 0.0) for m in metrics_list]
            ci_lower, ci_upper, mean = calculate_bootstrap_ci(pass_fail_values)
            report["task_performance"][category] = {
                "pass_rate_mean": mean,
                "pass_rate_ci_lower": ci_lower,
                "pass_rate_ci_upper": ci_upper,
                "num_runs": len(metrics_list)
            }
    
    return report


def compare_generations(data_store: DataStore, gen1: int, gen2: int) -> Dict[str, Any]:
    """
    Compare two generations using statistical tests.
    
    Args:
        data_store: DataStore instance
        gen1: First generation number
        gen2: Second generation number
        
    Returns:
        Comparison report
    """
    fitness1 = [f for _, f in data_store.get_fitness_by_generation(gen1, "weighted_fitness")]
    fitness2 = [f for _, f in data_store.get_fitness_by_generation(gen2, "weighted_fitness")]
    
    if not fitness1 or not fitness2:
        return {"error": "Insufficient data for comparison"}
    
    # Permutation test
    p_value, observed_diff = permutation_test(fitness1, fitness2)
    
    # Win/tie/loss analysis
    win_analysis = win_tie_loss_analysis(fitness1, fitness2)
    
    return {
        "generation1": gen1,
        "generation2": gen2,
        "generation1_mean": sum(fitness1) / len(fitness1),
        "generation2_mean": sum(fitness2) / len(fitness2),
        "observed_difference": observed_diff,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "win_tie_loss": win_analysis
    }


def generate_evolution_report(data_store: DataStore, 
                             start_generation: int = 0,
                             end_generation: int = None) -> Dict[str, Any]:
    """
    Generate a comprehensive report for an evolutionary run.
    
    Args:
        data_store: DataStore instance
        start_generation: Starting generation
        end_generation: Ending generation (None for all)
        
    Returns:
        Comprehensive evolution report
    """
    # Get all generations
    cursor = data_store.conn.cursor()
    if end_generation is None:
        cursor.execute("SELECT MAX(generation_id) FROM generations")
        row = cursor.fetchone()
        end_generation = row[0] if row and row[0] else start_generation
    
    generation_reports = []
    for gen in range(start_generation, end_generation + 1):
        report = generate_generation_report(data_store, gen)
        generation_reports.append(report)
    
    # Overall statistics
    all_fitness = []
    for gen in range(start_generation, end_generation + 1):
        fitness_scores = data_store.get_fitness_by_generation(gen, "weighted_fitness")
        all_fitness.extend([f for _, f in fitness_scores])
    
    overall_stats = {}
    if all_fitness:
        ci_lower, ci_upper, mean = calculate_bootstrap_ci(all_fitness)
        overall_stats = {
            "mean_fitness": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "best_fitness": max(all_fitness),
            "worst_fitness": min(all_fitness)
        }
    
    return {
        "start_generation": start_generation,
        "end_generation": end_generation,
        "overall_statistics": overall_stats,
        "generation_reports": generation_reports
    }


def main():
    parser = argparse.ArgumentParser(description="Generate evolution reports")
    parser.add_argument("--db", default="evoagentbench.db", help="Database path")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--generation", type=int, help="Report for specific generation")
    parser.add_argument("--compare", nargs=2, type=int, metavar=("GEN1", "GEN2"),
                       help="Compare two generations")
    parser.add_argument("--start-gen", type=int, default=0, help="Start generation")
    parser.add_argument("--end-gen", type=int, help="End generation")
    
    args = parser.parse_args()
    
    data_store = DataStore(args.db)
    
    try:
        if args.compare:
            report = compare_generations(data_store, args.compare[0], args.compare[1])
        elif args.generation is not None:
            report = generate_generation_report(data_store, args.generation)
        else:
            report = generate_evolution_report(
                data_store, args.start_gen, args.end_gen
            )
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))
    
    finally:
        data_store.close()


if __name__ == "__main__":
    main()
