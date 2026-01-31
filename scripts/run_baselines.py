"""
Run baseline agents against benchmark suite (Guide §5.9).

Evaluates Zero-Shot, ReAct, and optionally Previous Best baseline.
"""

import argparse

from evoagentbench.core.baseline_genomes import (
    create_zero_shot_baseline,
    create_react_baseline,
    create_previous_best_baseline,
)
from evoagentbench.core.orchestrator import EvoBenchOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run baseline agents")
    parser.add_argument("--db", default="evoagentbench.db", help="Database path")
    parser.add_argument("--benchmarks", default="benchmarks/v1.0", help="Benchmark suite path")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument(
        "--previous-best",
        action="store_true",
        default=True,
        help="Include Previous Best baseline from latest generation (default: True)",
    )
    parser.add_argument(
        "--no-previous-best",
        action="store_false",
        dest="previous_best",
        help="Skip Previous Best baseline",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help="Use this generation for Previous Best; default is latest",
    )
    args = parser.parse_args()

    orchestrator = EvoBenchOrchestrator(
        db_path=args.db,
        benchmark_path=args.benchmarks,
    )

    try:
        tasks = orchestrator.load_benchmark_suite()

        zero_shot = create_zero_shot_baseline()
        react = create_react_baseline()
        baselines = [
            ("zero_shot", zero_shot),
            ("react", react),
        ]

        if args.previous_best:
            gen = args.generation
            if gen is None:
                cur = orchestrator.data_store.conn.cursor()
                cur.execute("SELECT MAX(generation) FROM genomes")
                row = cur.fetchone()
                gen = row[0] if row and row[0] is not None else None
            if gen is not None:
                best_id = orchestrator.evolution_engine.get_best_genome(gen)
                if best_id:
                    best_genome = orchestrator.data_store.get_genome(best_id)
                    if best_genome:
                        prev_best = create_previous_best_baseline(best_genome)
                        baselines.append(("previous_best", prev_best))
                        print(f"Previous Best from generation {gen} (genome {best_id})")
                    else:
                        print("Warning: best genome not found, skipping Previous Best")
                else:
                    print("Warning: no best genome for generation, skipping Previous Best")
            else:
                print("Warning: no generations in DB, skipping Previous Best")
        
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
