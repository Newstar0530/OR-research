# Research Automation Agent

A local-first, human-in-the-loop research assistant for Industrial Engineering and Operations Research.

The project is organized around a reusable research engine:

```text
research goal
-> domain classification
-> literature indexing / novelty risk
-> gap synthesis
-> mathematical model draft
-> model critique
-> algorithm and method planning
-> experiment implementation
-> sandboxed execution and debugging
-> experiment contract validation
-> sensitivity analysis
-> cautious report draft
-> automated reviewer feedback
```

## What It Does

- Builds an inspectable run folder for every research attempt.
- Uses pydantic schemas for agent outputs and core contracts.
- Classifies OR/IE topics into reusable domain profiles.
- Indexes local PDF/DOCX literature without fabricating citations.
- Produces a generic problem schema, research protocol, method registry, metric registry, and artifact manifest.
- Generates reproducible MI 0-1 (binary) integer programming instances and **actually solves them**
  with CP-SAT, Pyomo/Gurobi/CBC/GLPK, exhaustive enumeration, and heuristics.
- Verifies every returned solution independently by substituting it back into every constraint,
  and records the recomputed objective rather than the one the solver printed.
- Establishes a *proved* optimum where possible and reports each method's relative gap to it.
- Runs generated Python experiments in a subprocess with timeouts and captured logs.
- Runs an AI-Scientist-style experiment workspace loop: copy workspace, mutate `experiment.py`, execute, plot, evaluate, and promote the best checkpoint.
- Records a best-first tree-search policy queue, workspace lineage, failed-node repair trace, plot aggregation, and LLM interaction log.
- Validates `results.csv` against a generic experiment contract.
- Drafts cautious reports and marks claims that require human verification.
- Writes a LaTeX paper draft and attempts local PDF compilation when `pdflatex` is available.
- Supports mock LLM mode for deterministic offline testing.
- Supports OpenAI-compatible and Gemini-style providers through `LLMClient`.

## What It Does Not Do

- It does not fully automate science.
- It does not prove novelty.
- It does not guarantee mathematical correctness.
- It does not automatically submit papers.
- It does not treat automated reviewer feedback as acceptance.
- It does not fabricate citations; unknown literature remains unknown.
- It does not claim an improvement that fails a significance test, even when the point estimate
  looks good.

## Ethical Warning

Do not submit AI-generated papers, reports, or claims without human verification and appropriate disclosure. All outputs are drafts for inspection, critique, and reproducibility work.

## Install

```powershell
cd C:\python\research_automation_agent
python -m pip install -e ".[dev]"
```

Optional modeling packages:

```powershell
python -m pip install -e ".[optimization]"
```

Optional Aider-based code editing backend:

```powershell
python -m pip install -e ".[code-editing]"
```

## Run The Generic Pipeline

```powershell
cd C:\python\research_automation_agent
python -m src.main run --config configs\default.yaml
```

Each run is written under:

```text
runs/{timestamp}_{project_name}/
```

To prepare a follow-up run from the previous run's readiness plan:

```powershell
python -m src.main prepare-next --run-dir runs\20260607_000933_generic_or_research_run
python -m src.main run --config runs\20260607_000933_generic_or_research_run\next_cycle_config.yaml
```

The generated config keeps the previous settings, applies `next_config_patch.yaml`,
increments the project name suffix, and records `previous_run_dir`.

To let the CLI continue automatically only when no human-required gate is present:

```powershell
python -m src.main prepare-next --run-dir runs\20260607_000933_generic_or_research_run --autorun-safe-only
```

If the run has unresolved literature, claim, or human-gate issues, the command writes
the next config but skips autorun and prints the reasons.

For bounded overnight-style cycles:

```powershell
python -m src.main run-cycles --config configs\default.yaml --max-cycles 3
```

Each cycle re-checks the previous run's safety plan. The loop stops when a P1
human-required issue appears.

## Available Experiment Templates

`experiment_template` in the config pins which scaffold is used, overriding topic classification:

| key | what it does |
| --- | --- |
| `binary_program` | Real MI 0-1 solving with exact references and verified solutions. |
| `generic_or` | Synthetic baseline-vs-proposed scaffold; plumbing only, not evidence. |
| `scheduling`, `inventory`, `network_analysis`, `stochastic_models` | Domain-specific synthetic scaffolds. |

Leave `experiment_template` empty to let the domain classifier choose.

## Configure Your Own Research Topic

Edit `configs/default.yaml` or create a new YAML file with:

```yaml
project_name: my_or_research
research_goal: "Your research topic or abstract here."
domain: "Industrial Engineering / Operations Research"
research_domain_mode: "auto"
template_path:
output_dir: "runs"
previous_run_dir:
max_ideas: 3
max_experiments: 3
max_debug_attempts: 2
solver_timeout_seconds: 60
llm_provider: "mock"
model_name: "mock-ie-or"
use_mock_llm: true
enable_novelty_check: true
enable_report_generation: true
literature_dir:
enable_semantic_scholar: false
enable_llm_agent_runtime: true
enable_autonomous_loop: true
max_research_iterations: 2
max_candidate_branches: 3
autonomous_patience: 2
min_metric_improvement: 0.0
primary_metric: "objective"
objective_direction: "minimize"
experiment_template:                   # e.g. binary_program; empty = classifier decides
primary_metric_method:                 # restrict the node metric to one method
baseline_method:                       # explicit comparison pair
proposed_method:
code_editing_backend: "deterministic"  # deterministic, aider, auto, disabled
aider_command: "aider"
aider_model:
aider_timeout_seconds: 120
```

If you have a starter experiment script, set `template_path` to it. The script should write `results.csv`, create figures under `figures/`, and print a final `SUMMARY_JSON:` line.

## Core Framework

The reusable framework lives in `src/core/`:

- `research_protocol.py`: inspectable staged research protocol.
- `problem_schema.py`: provider-neutral OR research problem schema.
- `benchmark_loader.py`: local benchmark file indexing for CSV, JSON, TXT, and DAT instances.
- `solve_result.py`: the `SolveResult` contract - solver status, dual bound, MIP gap, verification
  outcome, and distance to a proved optimum, in a stable `results.csv` column order.
- `binary_program.py`: reproducible 0-1 integer programs (knapsack, multidimensional knapsack,
  set cover) with explicit difficulty knobs.
- `solution_checker.py`: independent feasibility verification and solver-claim cross-examination.
- `solver_backends.py`: CP-SAT, Pyomo (Gurobi/HiGHS/CBC/GLPK/SCIP), exhaustive enumeration,
  greedy, and local search behind one verified `solve_instance()` entry point.
- `experiment_contract.py`: required experiment outputs and result validation, in two tiers
  (legacy, and a stricter solver tier detected automatically).
- `experiment_workspace.py`: AI-Scientist-style editable experiment workspace creation and copying.
- `template_registry.py`: maps problem families to reusable experiment scaffolds.
- `method_registry.py`: reusable exact, heuristic, simulation, and sensitivity method specs.
- `metric_registry.py`: common objective, runtime, gap, feasibility, and robustness metrics.
- `model_ir.py`: heuristic structured model representation for sets, parameters, variables, objectives, and constraints.
- `model_exporter.py`: Pyomo and OR-Tools skeleton exporters from the model IR.
- `solver_adapter.py`: local solver/tool capability detection and fallback recommendations.
- `experiment_search.py`: workspace-based autonomous experiment loop with best-first frontier tracking.
- `code_editing_backend.py`: pluggable failed-workspace repair backends, including deterministic repair and optional Aider repair.
- `workspace_debugger.py`: deterministic repair hooks for failed generated workspaces.
- `registry.py`: small explicit registry abstraction.

This is inspired by useful AI-Scientist-v2 design patterns such as journals, metrics, stage transitions, and checkpointed artifacts, but adapted to OR/IE and kept local-first.

## Solver-Backed MI 0-1 Experiments

The `generic_or` scaffold produces synthetic numbers: its "objective" is a formula in which the
proposed method is handed a positive `improvement` term, so it wins by construction. That is useful
only for exercising the pipeline plumbing.

The `binary_program` template is the opposite. For each `(family, size, difficulty, seed)` it

1. generates a real 0-1 integer program (`max/min c'x` subject to `Ax {<=,>=,==} b`, `x` binary),
2. establishes a **proved** optimum by exhaustive enumeration or by CP-SAT reporting `optimal`,
   and records `None` when no proof was obtained,
3. solves it with each configured method,
4. substitutes every returned solution back into every constraint independently,
5. writes the recomputed objective, the MIP gap against the dual bound, and the relative gap to the
   proved optimum.

So `proposed beats baseline` can come out either way, and a method that returns an infeasible
vector is recorded as a defect instead of a score.

```powershell
python -m pip install -e ".[optimization]"   # pyomo, ortools, scipy, networkx
python -m src.main run --config configs\binary_program.yaml
```

Extra artifacts from a solver-backed run:

- `verification_report.md`: rows, trustworthy rows, verification failures, instances with a proved optimum.
- `solver_backend_report.json`: which solvers were available on this machine, and why the others were not.
- `instances/*.json`: the exact instance data, so any result can be re-solved independently.
- `figures/gap_to_optimum_by_size.png`, `figures/runtime_by_size.png`.

### Verification Is Not Optional

`solve_instance()` never returns a record that has not been through `verify_solver_claim`. When a
backend claims a solution that fails verification, three things happen: the status is rewritten to
`error`, the contradiction is written into `notes`, and `verification_ok` becomes False. The row is
kept so the defect stays visible, but `is_trustworthy` is False, so no metric, comparison, or
statistic will average it in. The solver contract then fails the whole run.

A solver that reports a solution it cannot produce is a bug in the experiment, not a data point.

### Choosing The Node Metric

For solver-backed runs the node metric is set with three config keys:

```yaml
primary_metric: "gap_to_known_optimum"
objective_direction: "minimize"
primary_metric_method: "local_search"   # the method under test
baseline_method: "greedy"
proposed_method: "local_search"
```

`primary_metric_method` matters more than it looks. Without it, a node's metric is "whichever
method did best", which on a solver-backed run is always the exact reference solver at gap 0.0 -
so every node would score identically and the tree search would have nothing to rank.

### Statistical Claims

`statistical_evidence.md` reports Welch's t-test (SciPy) or a normal approximation, Cohen's d, and
an explicit `supports_improvement_claim` flag. A significance test that fails now **vetoes** the
optimistic labels: an observed improvement that cannot be distinguished from noise is reported as
`inconclusive_not_significant`, never as strong evidence. Only
`supports_improvement_claim: True` licenses a comparative claim in the write-up.

## Paper-Inspired Automation Pattern

The implementation follows a conservative version of the workflow described in
`Towards end-to-end automation of AI research`:

- Ideation remains structured and inspectable through `idea_archive.json` and `selected_idea.json`.
- Experimentation uses a workspace tree: each node has its own `experiment.py`, `plot.py`, prompt metadata, logs, results, and figures.
- The best successful checkpoint is promoted to `best_experiment.py`, `best_plot.py`, `best_prompt.json`, and `best_notes.md`.
- Failed experiment nodes are not ignored. The system records `repair_trace.md` and retries deterministic repairs when possible.
- Aider can be enabled as an optional code-editing backend for failed experiment nodes, while keeping edits restricted to the node workspace.
- Plot execution is separate from experiment execution, then summarized in `plot_aggregation.md`.
- Tree-search decisions are made visible in `bfts_frontier.md` and `bfts_policy_queue.md`.
- Write-up produces `final_report.md` and `final_paper.tex`; PDF compilation is attempted only when local LaTeX tooling is installed.
- Automated review is a draft review signal only. Human verification remains mandatory.

This is not a paper-submission machine. It is a local research workbench that keeps intermediate artifacts visible so a researcher can audit, revise, and rerun.

## Output Files

Typical run artifacts include:

- `research_protocol.md`
- `artifact_manifest.json`
- `run_status.json`
- `run_status.md`
- `problem_schema.json`
- `research_state.json`
- `llm_agent_trace.json`
- `llm_agent_trace.md`
- `method_registry.json`
- `metric_registry.json`
- `config_used.yaml`
- `domain_selection.json`
- `benchmark_manifest.json`
- `benchmark_manifest.md`
- `domain_tool_spec.json`
- `domain_tool_spec.md`
- `solver_tool_report.json`
- `solver_tool_report.md`
- `recommended_solver_tools.json`
- `domain_profile.md`
- `literature_index.csv`
- `literature_review.md`
- `idea_archive.json`
- `selected_idea.json`
- `novelty_report.md`
- `research_gap_analysis.md`
- `model_draft.md`
- `model_ir.json`
- `model_ir.md`
- `model_export_manifest.json`
- `model_export_pyomo.py`
- `model_export_ortools.py`
- `model_export_diagnostics.md`
- `model_critique.md`
- `algorithm_plan.md`
- `generated_experiment.py`
- `experiment_workspace/`
- `best_experiment.py`
- `best_prompt.json`
- `best_notes.md`
- `best_plot.py`
- `autonomous_journal.json`
- `autonomous_journal.md`
- `strategy_trace.json`
- `strategy_trace.md`
- `mutation_trace.json`
- `mutation_trace.md`
- `patch_trace.json`
- `patch_trace.md`
- `workspace_lineage.json`
- `workspace_lineage.md`
- `bfts_frontier.json`
- `bfts_frontier.md`
- `bfts_policy_queue.json`
- `bfts_policy_queue.md`
- `repair_trace.json`
- `repair_trace.md`
- `plot_aggregation.json`
- `plot_aggregation.md`
- `aggregate_figures/`
- `llm_interactions.json`
- `llm_interactions.md`
- `agent_feedback_rounds.json`
- `agent_feedback_rounds.md`
- `state_snapshots/`
- `evidence_synthesis.md`
- `ablation_plan.md`
- `autonomous_search/`
- `execution_log.json`
- `experiment_journal.md`
- `results.csv`
- `experiment_contract.md`
- `result_evaluation.md`
- `statistical_evidence.json`
- `statistical_evidence.md`
- `verification_report.md` (solver-backed runs)
- `solver_backend_report.json` (solver-backed runs)
- `research_tree.json`
- `research_tree.md`
- `sensitivity_report.md`
- `claim_check.md`
- `citation_report.json`
- `citation_report.md`
- `references.bib`
- `literature_grounding.json`
- `literature_grounding.md`
- `autonomy_readiness.json`
- `autonomy_readiness.md`
- `next_research_cycle_plan.json`
- `next_research_cycle_plan.md`
- `next_config_patch.yaml`
- `final_report.md`
- `final_paper.tex`
- `latex_compile_report.json`
- `latex_compile_report.md`
- `pdf_review_stub.json`
- `pdf_review_stub.md`
- `automated_review.md`
- `stage_order.json`
- `figures/`

`autonomy_readiness.md` is a gatekeeping summary for human researchers. It checks whether
the run produced enough inspectable evidence, baseline comparison, claim checks, and review
signals to be worth human review. It does not certify the research as correct or publishable.

`next_research_cycle_plan.md` converts readiness warnings into a prioritized follow-up plan.
Low-risk items may include a suggested `next_config_patch.yaml`; literature, claim, and human
gate issues remain explicitly human-required.

`literature_grounding.md` maps candidate report claims to snippets from `literature_index.csv`
when local literature is provided. It uses keyword overlap only, so grounded claims still need
manual citation verification.

`solver_tool_report.md` records which OR tools are installed locally, such as Pyomo, Gurobi,
OR-Tools, SciPy, and NetworkX, and lists fallback options.

`model_ir.md` is a structured extraction of the mathematical model draft. It is meant for
inspection and future solver export, not as proof that the model is correct.

`statistical_evidence.md` summarizes method-level means and confidence intervals, runs Welch's
t-test when SciPy is available (falling back to a normal approximation and saying so in
`test_used`), and reports Cohen's d. A non-significant result vetoes the optimistic evidence
labels. It is still unpaired testing on a specific instance set, not a substitute for a
domain-appropriate paired design.

`final_paper.tex` is a LaTeX draft assembled from the run artifacts. It preserves the
human-verification warning and should be edited before any formal use.

`benchmark_manifest.md` records local benchmark files and warns when experiments are using
synthetic data only.

`model_export_pyomo.py` and `model_export_ortools.py` are solver skeletons generated from
the model IR. They intentionally require human completion before solving.

`research_tree.md` summarizes the autonomous experiment nodes in a best-first-search style
tree and identifies next expansion candidates.

`experiment_workspace/` is the root editable workspace for the AI-Scientist-style loop.
Each search node copies a parent workspace, edits `experiment.py`, runs it, and records
lineage in `workspace_lineage.md`. The best workspace is promoted as `best_experiment.py`,
`best_prompt.json`, `best_notes.md`, and `best_plot.py`.

`bfts_frontier.md` records the current best-first frontier used to decide which successful
workspace states should be expanded next.

`bfts_policy_queue.md` records a scored priority queue for the autonomous experiment tree.
It combines node value, exploration pressure, and failure penalties so the search decision is
inspectable rather than hidden.

`repair_trace.md` records failed experiment nodes, any deterministic repairs that were applied,
and the retry result. It is intentionally conservative; complex semantic repairs still require
human review or a stronger code-editing backend.

To use Aider for failed experiment-node repair, install and configure Aider separately, then set:

```yaml
code_editing_backend: "aider"  # or "auto" to try deterministic repair first
aider_command: "aider"
aider_model: "your-aider-model"  # optional; can also rely on Aider environment config
aider_timeout_seconds: 120
```

The Aider backend writes `aider_repair_prompt_attempt_*.md` inside the failed node workspace.
It asks Aider to edit only `experiment.py` and preserve the experiment output contract.

`plot_aggregation.md` lists each node's plot execution status and collected figures. Node-level
figures are also copied into `aggregate_figures/` for quick inspection.

`llm_interactions.md` records provider/model metadata, prompt and response previews, timing, and
approximate token counts for agent calls. It is for auditability and cost/debug analysis.

`latex_compile_report.md` records whether `final_paper.tex` was compiled. If `pdflatex` is not
installed, it reports `skipped_no_pdflatex` instead of failing the run.

`pdf_review_stub.md` performs lightweight structural checks on the LaTeX/PDF output. It is not a
visual paper review and does not judge scientific correctness.

`citation_report.md` and `references.bib` are generated from local literature metadata.
They use filename/title guesses and must be corrected before formal writing.

`run_status.md` records completed stages, runtime, and approximate artifact-token counts.
If a run is interrupted, use:

```powershell
python -m src.main resume --run-dir runs\your_incomplete_run
```

## Real LLM API

Set credentials through environment variables. Do not put API keys in source code.

OpenAI-compatible:

```powershell
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

Gemini:

```powershell
$env:GEMINI_API_KEY="your_key"
```

Then set:

```yaml
use_mock_llm: false
llm_provider: "openai-compatible"  # or "gemini"
model_name: "your-model-name"
```

## Add A Reusable OR Capability

Prefer adding reusable capabilities instead of one-off demos:

1. Add a domain profile in `domain_profiles/` only if it represents a broad research family.
2. Add a generic scaffold in `templates/` only if it can serve many topics in that family.
3. Register it in `src/core/template_registry.py`.
4. Add method specs in `src/core/method_registry.py` when they are reusable.
5. Add metrics in `src/core/metric_registry.py` when they are broadly meaningful.
6. Add tests for the contract, registry, and orchestration behavior.

## Run Tests

```powershell
cd C:\python\research_automation_agent
python -m pytest
```

## Known Gaps

Honest list of what is still missing, in priority order:

- The tree-search policy in `bfts_policy_queue.md` is still computed *after* the loop, so it
  reports the search rather than steering it.
- Failure repair is deterministic string editing; there is no traceback-to-LLM rewrite loop.
- Mutation is exact string replacement, so it only works on templates that expose the expected
  constant names.
- No figure is ever looked at; a blank or non-converged plot is reported as success.
- No transformation-equivalence verification yet (bilevel to KKT to MI 0-1 to QUBO), which is the
  one correctness check OR can do and machine-learning pipelines cannot.
- Cross-run memory is written but never read back, so refuted hypotheses are not remembered.

## Limitations

- The system can scaffold and run initial computational research, but it cannot replace mathematical proof or expert review.
- Literature extraction is metadata-light; citation details must be verified by a human.
- The default experiment scaffold is intentionally generic.
- Strong research claims require domain-specific validation, statistical checks, and advisor/reviewer inspection.
