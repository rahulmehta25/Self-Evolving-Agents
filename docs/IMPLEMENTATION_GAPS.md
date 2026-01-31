# Implementation Gaps vs. EvoAgentBench Guide (Manus)

This document lists what the technical implementation guide specifies but is **not yet implemented** or only **partially implemented** in the current codebase. Items are grouped by section of the guide.

---

## 4. Benchmark Suite

### 4.3 Input/Output Contracts — ✅ Done
- **Task `input_params` → prompt:** Implemented in `agent_runner.run_task` via `format_map(defaultdict(str, input_params))`.

### 4.5 Dataset Versioning and Governance — ✅ Done (code parts)
- **benchmark_suite_commit:** Orchestrator `_get_benchmark_suite_commit()` runs `git rev-parse HEAD` in benchmark dir; stored in run manifest.
- **Governance:** PR/approval process is organizational, not encoded in code.

---

## 5. Evaluation Protocol

### 5.1 Run Orchestration — Partial
- **LLM seed:** ✅ Vertex Gemini receives `seed` in `GenerationConfig`.
- **NumPy RNG:** ✅ `np.random.seed(run_seed)` set in agent runner when numpy is available.
- **Environment pinning:** `dependency_hashes` / `tool_versions` still placeholders.

### 5.2 Controlled Execution — Partial
- **Time budget:** Still loop-based check only; no thread/signal timeout.
- **Stochastic retries:** ✅ Vertex LLM adapter has 3 tries + exponential backoff; raises `ExternalFailureError` on exhaustion; runner sets status `EXTERNAL_FAILURE`.

### 5.4 Judge Design — Gaps
- **Order bias:** Guide says the order of agent outputs (when comparing) should be **randomized**. Judge evaluation does not implement this.
- **Calibration suite integration:** A **Calibration Suite** with expert-scored outputs and drift checks is specified. There is a `CalibrationSuite`/`check_calibration` and a `calculate_cohens_kappa` stub, but **no wiring** into the normal evaluation path (e.g. “run calibration every N runs” or “fail CI if drift &gt; threshold”).
- **Inter-judge agreement:** Guide requires **two judge models** and **Cohen’s Kappa** for high-stakes evaluations. Only a simplified Kappa-style helper exists; **no dual-judge scoring path** is implemented.
- **Structured JSON output:** Judge is required to return structured JSON (scores + justification). The mock does; **real judge calls** must enforce a fixed schema (e.g. Pydantic or JSON schema).

### 5.5 Citation Verification — ✅ Done (checker)
- **checkers/citation.py:** Implemented `check_citation_accuracy` with source existence, claim/snippet checks, and `citation_fidelity` per guide. RAG/tool layer logging of snippets/source/span remains for when real retrieval tools exist.

### 5.6 Scoring Aggregation — ✅ Done
- **Weights in EvolutionEngine config:** `EvolutionEngine(..., metric_weights=...)`; orchestrator passes `evolution_engine.metric_weights` into `evaluator.compute_weighted_fitness`.

### 5.7 Statistical Reporting — Partial
- **Win/Tie/Loss by task:** Guide asks for “percentage of **tasks** where Agent A significantly outperformed Agent B” (and tie/loss). Current `win_tie_loss_analysis` works on **per-run fitness vectors**, not per-task outcomes. So we do not yet report “% of tasks won/tied/lost” with significance.
- **Fix:** Support per-task outcomes (e.g. pass_fail or task-level score) and add a report that computes win/tie/loss over tasks with a chosen significance rule.

### 5.8 Ablation Framework — Partial
- **Ablation config file:** Guide describes a **config file** with `base_genome_id`, `factor_key`, `factor_values`. Ablations are driven by CLI flags in `run_ablation.py`, **not** by a config file.
- **Pairwise significance between factor values:** Guide requires “pair-wise significance tests between the factor values.” The ablation script evaluates each factor value but **does not** run pairwise permutation (or similar) tests between them or add those to the output.
- **Fix:** Add support for an ablation config file and a small stats step that runs pairwise tests between factor levels and attaches results to the ablation report.

### 5.9 Baseline Agents — ✅ Done
- **Previous Best baseline:** `run_baselines.py` includes Previous Best by default (`--previous-best`), loads best genome from latest generation (or `--generation N`), and evaluates it with Zero-Shot and ReAct.

---

## 6. Reproducibility and Deterministic Replay

### 6.1 Run Manifest — ✅ Done (prompt_hashes, benchmark_suite_commit)
- **prompt_hashes:** Computed and stored in manifest. **dependency_hashes / tool_versions:** still placeholders.

### 6.4 Artifact Bundles — ✅ Done
- **export_artifact_bundle** in `core/artifact_bundle.py` builds the zip (all 6 elements). CLI `--export-artifact RUN_ID`; optional `--gcs-bucket` / `EVOAGENTBENCH_GCS_BUCKET` for upload.

---

## 7. Data Storage Layer

### 7.1 Schema — Partial
- **Genomes: schema_version:** v2 hook in the guide expects a `schema_version` column on `genomes` for typed validation. **Not present** in the current schema.
- **DuckDB:** Guide prefers DuckDB; implementation uses **SQLite** only. No blocking gap, but DuckDB is not wired in.
- **tasks.category:** Stored as a single string (e.g. comma-separated). Guide models tasks with **list[string]** categories; this is representable but any “list” semantics (e.g. “all tasks in category X”) must match that.

---

## 8. Evolution Loop

### 8.3 Overfitting / Benchmark Hacking — ✅ Done
- **Holdout designation:** Tasks are sorted by `(task_id, version)` and the last `holdout_fraction` are holdout; same split every run. Main tasks used for fitness; holdout for reporting only.

---

## 9. V1 Hooks for V2

### GenomeValidator (Section 9 table) — ✅ Done
- **evoagentbench.core.genome_validator:** `validate_genome(genome)` returns `(ok, err)`. Used in orchestrator `initialize_genomes` before save.

### Evaluator pre_run_check (Section 9 table) — ✅ Done
- **Evaluator.pre_run_check(task, genome, budget):** Validates budget fields and required task/genome keys. Called in orchestrator `evaluate_genome` before running the agent.

---

## 10–13. Reference Plan, Pseudocode, Quality Checklist

- **Repository structure:** Matches the guide except `checkers/citation.py` is **missing** (see 5.5).
- **Quality checklists (Section 13):** The guide’s reproducibility, evaluation-validity, and failure-mode checklists are **not** automated. Implementing the above gaps would move the code toward satisfying those checklists; consider adding a small “compliance” script or test that checks trace integrity, manifest completeness, holdout usage, etc.

---

## Summary Table

| Area | Gap | Priority (if aligning to guide) |
|------|-----|----------------------------------|
| Benchmark | Inject `input_params` into `prompt_template` | High |
| Benchmark | Capture and store `benchmark_suite_commit` in manifest | High |
| Run | Seed NumPy RNG; pass seed to real LLM when integrated | High |
| Run | Time budget via thread/signal timeout | Medium |
| Run | Retries + `ExternalFailure` for external failures | Medium |
| Run | Prompt/dependency/tool hashes and versions in manifest | Medium |
| Judge | Order bias randomization; calibration in the loop | Medium |
| Judge | Dual-judge + Cohen’s Kappa for high-stakes runs | Low (v1.5) |
| Citation | Add `checkers/citation.py`; trace-based source/span + MCS/fuzzy | High |
| Citation | RAG/tool layer logging snippets and source/span | High when RAG exists |
| Reporting | Win/Tie/Loss by task with significance | Medium |
| Ablation | Config file; pairwise significance between factor values | Medium |
| Baselines | Run “Previous Best” in `run_baselines.py` | Medium |
| Artifacts | Artifact bundle zip + GCS upload | Done |
| Evolution | Designated holdout set (stable task set) | Done |
| Weights | Versioned weights in evolution/config | Done |
| V2 hooks | GenomeValidator module | Done |
| V2 hooks | Evaluator.pre_run_check | Done |

---

*Generated from a review of the EvoAgentBench Guide (Manus) against the current codebase.*
