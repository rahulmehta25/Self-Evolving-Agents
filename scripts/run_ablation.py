"""
Run ablation studies to test the impact of specific genome parameters.

Implements the ablation framework from the guide.
"""

import argparse
import json
import uuid
from typing import Any, Dict, List

from evoagentbench.core.data_store import DataStore
from evoagentbench.core.orchestrator import EvoBenchOrchestrator


def run_ablation(orchestrator: EvoBenchOrchestrator,
                 base_genome_id: str,
                 factor_key: str,
                 factor_values: List[Any]) -> Dict[str, Any]:
    """
    Run an ablation study.
    
    Args:
        orchestrator: EvoBenchOrchestrator instance
        base_genome_id: ID of the base genome
        factor_key: Key of the factor to vary (e.g., "llm_config.temperature")
        factor_values: List of values to test
        
    Returns:
        Ablation results
    """
    # Get base genome
    base_genome = orchestrator.data_store.get_genome(base_genome_id)
    if not base_genome:
        raise ValueError(f"Base genome {base_genome_id} not found")
    
    # Load tasks
    tasks = orchestrator.load_benchmark_suite()
    
    # Create ablation genomes
    ablation_genomes = []
    for value in factor_values:
        import copy
        genome = copy.deepcopy(base_genome)
        genome["genome_id"] = str(uuid.uuid4())
        genome["generation"] = -1  # Special generation for ablations
        
        # Set factor value (supports nested keys like "llm_config.temperature")
        keys = factor_key.split(".")
        target = genome
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        
        genome_id = orchestrator.data_store.save_genome(genome)
        ablation_genomes.append({
            "genome_id": genome_id,
            "factor_value": value
        })
        
        # Save ablation record
        ablation_id = str(uuid.uuid4())
        orchestrator.data_store.conn.cursor().execute("""
            INSERT INTO ablations (ablation_id, base_genome_id, factor_key, factor_value)
            VALUES (?, ?, ?, ?)
        """, (ablation_id, base_genome_id, factor_key, str(value)))
        orchestrator.data_store.conn.commit()
    
    # Evaluate each ablation genome
    results = []
    for ablation_genome in ablation_genomes:
        genome_id = ablation_genome["genome_id"]
        factor_value = ablation_genome["factor_value"]
        
        print(f"Evaluating {factor_key} = {factor_value}...")
        
        # Evaluate against all tasks
        ablation_metrics = []
        for task in tasks:
            run_seed = hash(f"{genome_id}_{task['task_id']}") % (2**31)
            result = orchestrator.evaluate_genome(genome_id, task, run_seed)
            ablation_metrics.append(result["metrics"])
        
        # Aggregate metrics
        if ablation_metrics:
            avg_fitness = sum(m.get("weighted_fitness", 0.0) for m in ablation_metrics) / len(ablation_metrics)
            avg_pass_fail = sum(m.get("pass_fail", 0.0) for m in ablation_metrics) / len(ablation_metrics)
        else:
            avg_fitness = 0.0
            avg_pass_fail = 0.0
        
        results.append({
            "factor_value": factor_value,
            "genome_id": genome_id,
            "avg_fitness": avg_fitness,
            "avg_pass_fail": avg_pass_fail,
            "metrics": ablation_metrics
        })
    
    return {
        "base_genome_id": base_genome_id,
        "factor_key": factor_key,
        "factor_values": factor_values,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--db", default="evoagentbench.db", help="Database path")
    parser.add_argument("--base-genome", required=True, help="Base genome ID")
    parser.add_argument("--factor", required=True, help="Factor key (e.g., llm_config.temperature)")
    parser.add_argument("--values", required=True, nargs="+", 
                       help="Factor values to test")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    orchestrator = EvoBenchOrchestrator(db_path=args.db)
    
    try:
        # Parse values (try to convert to appropriate types)
        factor_values = []
        for val in args.values:
            try:
                if '.' in val:
                    factor_values.append(float(val))
                else:
                    factor_values.append(int(val))
            except ValueError:
                factor_values.append(val)
        
        results = run_ablation(
            orchestrator,
            args.base_genome,
            args.factor,
            factor_values
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Ablation results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
    
    finally:
        orchestrator.data_store.close()


if __name__ == "__main__":
    main()
