"""
Run baseline agents against benchmark suite.

Evaluates the three baseline agents for comparison.
"""

import argparse

from evoagentbench.core.baseline_genomes import (
    create_zero_shot_baseline,
    create_react_baseline
)
from evoagentbench.core.orchestrator import EvoBenchOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run baseline agents")
    parser.add_argument("--db", default="evoagentbench.db", help="Database path")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    orchestrator = EvoBenchOrchestrator(
        db_path=args.db,
        benchmark_path="benchmarks/v1.0"
    )
    
    try:
        # Load tasks
        tasks = orchestrator.load_benchmark_suite()
        
        # Create baseline genomes
        zero_shot = create_zero_shot_baseline()
        react = create_react_baseline()
        
        baselines = [
            ("zero_shot", zero_shot),
            ("react", react)
        ]
        
        results = {}
        
        for baseline_name, genome in baselines:
            print(f"Evaluating {baseline_name} baseline...")
            genome_id = orchestrator.data_store.save_genome(genome)
            
            baseline_results = []
            for task in tasks:
                run_seed = hash(f"{genome_id}_{task['task_id']}") % (2**31)
                result = orchestrator.evaluate_genome(genome_id, task, run_seed)
                baseline_results.append(result)
            
            results[baseline_name] = {
                "genome_id": genome_id,
                "results": baseline_results
            }
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            import json
            print(json.dumps(results, indent=2))
    
    finally:
        orchestrator.data_store.close()


if __name__ == "__main__":
    main()
