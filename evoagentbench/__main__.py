"""
CLI entry point for EvoAgentBench.

Run with: python -m evoagentbench

Starts a quick evolution run with default settings, or shows usage.
"""

import argparse
import sys

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        description="EvoAgentBench: Evolutionary optimization for LLM agents"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"EvoAgentBench {__version__}",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run a quick evolution (2 generations, 4 genomes)",
    )
    parser.add_argument(
        "--db",
        default="evoagentbench.db",
        help="Database path (default: evoagentbench.db)",
    )
    parser.add_argument(
        "--benchmarks",
        default="benchmarks/v1.0",
        help="Benchmark suite path (default: benchmarks/v1.0)",
    )
    args = parser.parse_args()

    if args.run:
        from .core.orchestrator import EvoBenchOrchestrator
        from .core.baseline_genomes import create_react_baseline

        print(f"EvoAgentBench {__version__} — quick run")
        orch = EvoBenchOrchestrator(db_path=args.db, benchmark_path=args.benchmarks)
        genome = create_react_baseline()
        genome["genome_id"] = "quick_run_baseline"
        genome["generation"] = 0
        initial = [genome]
        try:
            results = orch.run_evolution(
                num_generations=2,
                initial_genomes=initial,
                population_size=4,
                holdout_fraction=0.1,
            )
            print(f"Done. Best from final gen: {results['history'][-1].get('best_genome_id', 'N/A')}")
        finally:
            orch.data_store.close()
        return 0

    parser.print_help()
    print("\nExample: python -m evoagentbench --run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
