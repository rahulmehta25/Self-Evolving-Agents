"""
Basic example: Run a simple evolutionary optimization.

This example demonstrates how to:
1. Initialize genomes
2. Run evolution
3. Generate reports
"""

from evoagentbench.core.orchestrator import EvoBenchOrchestrator


def main():
    # Initialize orchestrator
    orchestrator = EvoBenchOrchestrator(
        db_path="example.db",
        benchmark_path="benchmarks/v1.0"
    )
    
    # Define initial genomes
    initial_genomes = [
        {
            "genome_id": "baseline_v1",
            "system_prompt": "You are a helpful AI assistant. Answer questions accurately and concisely.",
            "llm_config": {
                "model_name": "gpt-4",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1000
            },
            "tools": [],
            "planner_type": "react",
            "generation": 0
        },
        {
            "genome_id": "creative_v1",
            "system_prompt": "You are a creative AI assistant. Think step by step and provide detailed answers.",
            "llm_config": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1500
            },
            "tools": [],
            "planner_type": "react",
            "generation": 0
        }
    ]
    
    print("Starting evolutionary optimization...")
    print(f"Initial population: {len(initial_genomes)} genomes")
    
    # Run evolution
    results = orchestrator.run_evolution(
        num_generations=3,
        initial_genomes=initial_genomes,
        population_size=10,
        holdout_fraction=0.1
    )
    
    print("\nEvolution completed!")
    print(f"Generations: {results['num_generations']}")
    
    # Print summary
    for gen_report in results["history"]:
        print(f"\nGeneration {gen_report['generation']}:")
        print(f"  Average fitness: {gen_report['avg_fitness']:.4f}")
        print(f"  Best genome: {gen_report['best_genome_id']}")
    
    # Close database
    orchestrator.data_store.close()


if __name__ == "__main__":
    main()
