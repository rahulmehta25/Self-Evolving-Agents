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
    parser.add_argument(
        "--export-artifact",
        metavar="RUN_ID",
        help="Export artifact bundle zip for run RUN_ID (Guide §6.4)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for --export-artifact (default: current directory)",
    )
    parser.add_argument(
        "--gcs-bucket",
        metavar="BUCKET",
        help="GCS bucket to upload artifact bundle to (or set EVOAGENTBENCH_GCS_BUCKET)",
    )
    parser.add_argument(
        "--llm",
        choices=["mock", "vertex"],
        default=None,
        help="Agent LLM: vertex=Vertex Gemini, mock=no API. Default: vertex if GCP/Vertex SDK available",
    )
    parser.add_argument(
        "--judge",
        choices=["mock", "vertex"],
        default="vertex",
        help="Judge LLM for soft metrics: vertex=Vertex Gemini, mock=placeholder scores. Default: vertex (set EVOAGENTBENCH_JUDGE_LLM=vertex to force)",
    )
    args = parser.parse_args()

    if args.export_artifact:
        import os
        from pathlib import Path
        from .core.data_store import DataStore
        from .core.artifact_bundle import export_artifact_bundle
        store = DataStore(args.db)
        try:
            bucket = args.gcs_bucket or os.environ.get("EVOAGENTBENCH_GCS_BUCKET", "").strip() or None
            path, gs_uri = export_artifact_bundle(
                store, args.export_artifact, Path(args.out_dir), gcs_bucket=bucket
            )
            print(f"Exported: {path}")
            if gs_uri:
                print(f"Uploaded: {gs_uri}")
        finally:
            store.close()
        return 0

    if args.run:
        import os
        from .core.orchestrator import EvoBenchOrchestrator
        from .core.baseline_genomes import create_react_baseline

        if args.llm is not None:
            os.environ["EVOAGENTBENCH_LLM"] = args.llm
        # Judge defaults to vertex; env is set so evaluator uses Vertex Gemini
        os.environ["EVOAGENTBENCH_JUDGE_LLM"] = args.judge
        from evoagentbench.runner import llm_adapters
        effective = llm_adapters.get_llm_provider()
        print(f"EvoAgentBench {__version__} — quick run (llm={effective})")
        orch = EvoBenchOrchestrator(db_path=args.db, benchmark_path=args.benchmarks, llm_provider=args.llm)
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
