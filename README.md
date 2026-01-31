# EvoAgentBench

A Reproducible, Benchmark-Driven Evolutionary Optimization Platform for LLM Agents

## Overview

EvoAgentBench is designed to bring **scientific rigor and evolutionary optimization** to the development of Large Language Model (LLM) agents. The platform provides a deterministic, reproducible, and auditable system for evaluating agent "genomes" (configurations of prompts, tools, memory, and planners) against a fixed, versioned benchmark suite.

## Features

- **Deterministic Evaluation**: Fully reproducible runs with comprehensive tracing
- **Evolutionary Optimization**: Tournament selection, mutation, and crossover operators
- **Multi-Metric Fitness**: Hard metrics (regex, JSON schema, Python unit tests) and soft metrics (LLM-as-Judge)
- **Statistical Rigor**: Bootstrap confidence intervals and significance testing
- **Reproducibility**: Complete trace logging with input/output hashing for deterministic replay
- **Benchmark Suite**: Versioned, immutable task specifications

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize a Database

```python
from evoagentbench.core.data_store import DataStore

data_store = DataStore("evoagentbench.db")
```

### 2. Load Benchmark Suite

```python
from evoagentbench.core.orchestrator import EvoBenchOrchestrator

orchestrator = EvoBenchOrchestrator(
    db_path="evoagentbench.db",
    benchmark_path="benchmarks/v1.0"
)

tasks = orchestrator.load_benchmark_suite()
```

### 3. Define Initial Genomes

```python
initial_genomes = [
    {
        "genome_id": "baseline_v1",
        "system_prompt": "You are a helpful AI assistant.",
        "llm_config": {
            "model_name": "gpt-4",
            "temperature": 0.0,
            "top_p": 1.0
        },
        "tools": [],
        "planner_type": "react"
    }
]
```

### 4. Run Evolution

```python
results = orchestrator.run_evolution(
    num_generations=10,
    initial_genomes=initial_genomes,
    population_size=20
)
```

### 5. Generate Reports

```bash
python scripts/generate_report.py --db evoagentbench.db --output report.json
```

## Using GCP (Vertex AI & Cloud Storage)

### Vertex AI Gemini (real LLM)

[Vertex AI Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs) is the **default** when the Vertex SDK is installed and GCP is configured. No flag needed.

1. **Auth**: `gcloud auth application-default login` and set your project (`gcloud config set project YOUR_PROJECT`).
2. **Env** (optional): `EVOAGENTBENCH_GCP_PROJECT=your-project`, `EVOAGENTBENCH_GCP_LOCATION=us-central1`.
3. **CLI**:  
   `python -m evoagentbench --run`  
   uses Vertex by default. To force mock (no API calls): `--llm mock`.  
   In code: `EvoBenchOrchestrator(...)` uses Vertex when available; pass `llm_provider="mock"` to disable.

Genome `llm_config.model_name` maps to Gemini (e.g. `gemini-1.5-flash`). The $300 GCP free trial applies when using Vertex AI.

### Judge (LLM-as-Judge) with Vertex

Use Vertex Gemini for soft metrics (coherence, completeness, correctness) instead of the mock:

- **Env:** `EVOAGENTBENCH_JUDGE_LLM=vertex`
- **CLI:** `python -m evoagentbench --run --judge vertex` (same ADC/project as agent)

Judge stays mock by default so you only pay for agent calls until you opt in.

### Cloud Storage (artifact bundles)

Upload artifact bundles to a GCS bucket:

1. **Create a bucket** in your project (e.g. `your-project-evoagentbench-artifacts`).
2. **Env**: `EVOAGENTBENCH_GCS_BUCKET=your-bucket-name`
3. **CLI**:  
   `python -m evoagentbench --export-artifact RUN_ID --out-dir ./out --gcs-bucket your-bucket-name`  
   The zip is written locally and to `gs://your-bucket-name/artifacts/artifact_bundle_*.zip`.

## Architecture

The system consists of five main components:

1. **Orchestrator**: Manages the overall workflow and coordinates components
2. **Evolution Engine**: Implements selection, mutation, and crossover
3. **Agent Runner**: Executes agent genomes against tasks with full tracing
4. **Evaluator**: Computes hard and soft metrics
5. **Data Store**: Central repository using SQLite/DuckDB

## Project Structure

```
EvoAgentBench/
├── evoagentbench/
│   ├── core/
│   │   ├── orchestrator.py      # Main entry point
│   │   ├── evolution_engine.py  # Selection, mutation logic
│   │   ├── genome_mutator.py    # Mutation operators
│   │   └── data_store.py        # Data access layer
│   ├── runner/
│   │   ├── agent_runner.py      # Executes agent, logs trace
│   │   ├── replay_runner.py     # Deterministic replay
│   │   └── trace_logger.py     # Trace logging
│   ├── evaluation/
│   │   ├── evaluator.py         # Hard metrics
│   │   ├── judge_evaluator.py   # LLM-as-Judge
│   │   └── stats_module.py      # Statistical analysis
│   └── schemas/
│       ├── task_spec_v1.json
│       ├── genome_spec_v1.json
│       └── trace_event_v1.json
├── benchmarks/
│   └── v1.0/                    # Versioned benchmark suite
├── checkers/                    # Custom Python checkers
└── scripts/
    ├── generate_report.py
    └── run_ablation.py
```

## Benchmark Tasks

Tasks are defined using JSON/YAML schemas with:
- Task ID and version
- Category (tool-use, retrieval/citation, code, planning, adversarial/injection)
- Prompt template and context
- Gold answers and checker configuration
- Budget constraints

## Evaluation Metrics

### Hard Metrics
- `pass_fail`: Binary task completion (0 or 1)
- `citation_fidelity`: Citation accuracy (0-1)
- `token_count`: Token usage
- `latency_seconds`: Execution time

### Soft Metrics (LLM-as-Judge)
- `coherence_score`: Response coherence (1-5)
- `completeness_score`: Task completeness (1-5)
- `correctness_score`: Information correctness (1-5)

### Weighted Fitness
Default weights:
- `pass_fail`: 0.5
- `citation_fidelity`: 0.3
- `coherence_score`: 0.1
- `latency_seconds`: -0.1 (penalty)

## Reproducibility

Every run is fully reproducible through:
- **Run Manifests**: Complete environment and configuration capture
- **Trace Logging**: Every LLM call and tool execution with input/output hashes
- **Replay Mode**: Deterministic replay using stored traces with hash verification

## Ablation Studies

Test the impact of specific parameters:

```bash
python scripts/run_ablation.py \
    --base-genome baseline_v1 \
    --factor llm_config.temperature \
    --values 0.0 0.3 0.7 1.0
```

## Statistical Reporting

Generate comprehensive reports with confidence intervals:

```bash
# Generation report
python scripts/generate_report.py --generation 5

# Compare generations
python scripts/generate_report.py --compare 0 5

# Full evolution report
python scripts/generate_report.py --start-gen 0 --end-gen 10
```

## Development Status

- ✅ M1: Data Foundation (Data Store, Schemas, Basic Agent Runner)
- ✅ M2: Evaluation Core (Evaluator, Judge Evaluator, Statistics)
- ✅ M3: Reproducibility (Replay Runner)
- ✅ M4: Evolution Loop (Evolution Engine, Genome Mutator, Orchestrator)
- ✅ M5: Reporting (Statistical Reports, Ablation Scripts)

## Future Work (v2.0)

- Full LLM API integration (currently mocked)
- Docker/Firecracker sandbox isolation
- Multi-objective optimization with Pareto selection
- Advanced mutation operators (LLM-driven)
- Safety suite gating
- Real-time observability UI

## License

MIT License

## Citation

If you use EvoAgentBench in your research, please cite:

```
EvoAgentBench: A Reproducible, Benchmark-Driven Evolutionary Optimization Platform for LLM Agents
```

## Contributing

Contributions are welcome! Please see the implementation guide for architecture details.
