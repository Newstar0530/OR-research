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
- Checks that a **transformation chain preserves the optimum**: bilevel -> KKT one-level ->
  MI 0-1, each step compared against an oracle that uses neither KKT nor Big-M.
- Derives Big-M constants that are *justified* rather than guessed, and locates the threshold
  below which the linearisation silently stops being exact.
- Carries the chain the last hop to a **QUBO**, separating the cost of discretisation from the
  correctness of the penalty, and refusing to call a sampler's best draw a ground state.
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
| `bilevel_transformation` | Verifies the bilevel -> KKT -> MI 0-1 chain against an independent oracle and locates the Big-M threshold. |
| `qubo_transformation` | Verifies the MI 0-1 -> QUBO encoding, separating discretisation cost from penalty correctness, and reports the bit cost. |
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
- `mixed_integer_program.py`: mixed continuous/binary models, solved by HiGHS, by Pyomo, or by
  exhaustive binary enumeration -- with the last used to cross-check the first.
- `bilevel.py`: linear bilevel programs with compact box bounds, and reproducible generators.
- `bilevel_oracle.py`: the independent ground truth (vertex enumeration plus a direct solve of
  the follower's LP). Uses no KKT, no complementarity and no Big-M, by design.
- `transformations.py`: `to_kkt_one_level`, `to_mi01`, `derive_big_m_bounds`, and an exact
  Big-M-free solver for the complementarity program.
- `equivalence.py`: per-step equivalence verdicts and the Big-M sensitivity sweep.
- `qubo.py`: QUBO models, exhaustive ground-state proof, a seeded annealer standing in for a
  sampler, and the Ising conversion hardware actually consumes.
- `qubo_transformations.py`: tight binary expansion, the binary-grid model, slack registers,
  penalty folding, and `derive_penalty_bound`.
- `qubo_equivalence.py`: three-way verification and the penalty sensitivity sweep.
- `experiment_contract.py`: required experiment outputs and result validation, in two tiers
  (legacy, and a stricter solver tier detected automatically).
- `experiment_workspace.py`: AI-Scientist-style editable experiment workspace creation and copying.
- `template_registry.py`: maps problem families to reusable experiment scaffolds.
- `method_registry.py`: reusable exact, heuristic, simulation, and sensitivity method specs.
- `metric_registry.py`: common objective, runtime, gap, feasibility, and robustness metrics.
- `model_ir.py`: heuristic structured model representation for sets, parameters, variables, objectives, and constraints.
- `model_exporter.py`: Pyomo and OR-Tools skeleton exporters from the model IR.
- `solver_adapter.py`: local solver/tool capability detection and fallback recommendations.
- `../llm_errors.py`: how a failed model call is classified, and whether retrying could help.
- `experiment_search.py`: the autonomous experiment loop. Each branch slot asks the search
  policy for its parent, so the run builds a real tree rather than a chain.
- `search_policy.py`: the in-loop draft / debug / improve decision, with the candidates it
  weighed written out per slot.
- `artifact_provenance.py`: where each artifact's content came from, and the `NOT GENERATED`
  block a stage emits instead of filler.
- `model_draft_inspection.py`: the one check that decides whether a formulation is specific
  enough to implement -- used to accept a draft and to critique it.
- `code_editing_backend.py`: pluggable failed-workspace repair backends -- a bounded deterministic
  rule, a guarded whole-file LLM rewrite, and optional Aider.
- `generated_code_guard.py`: what has to be true of a model-written script before it is allowed
  to replace a working one, and what makes a variant a variant rather than a copy.
- `workspace_debugger.py`: deterministic repair hooks for failed generated workspaces.
- `../execution/traceback_parser.py`: turns a subprocess traceback into a structured repair
  context -- framework frames removed, the failing line quoted, prior attempts listed.
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

`statistical_evidence.md` reports the test, the effect size, and an explicit
`supports_improvement_claim` flag. A significance test that fails **vetoes** the optimistic labels:
an observed improvement that cannot be distinguished from noise is reported as
`inconclusive_not_significant`, never as strong evidence. Only `supports_improvement_claim: True`
licenses a comparative claim in the write-up.

#### The Design Chooses The Test, Before The Test Is Run

An OR benchmark solves *the same instances* with every method, which makes the observations paired.
Handing paired data to an unpaired test is not a stylistic choice: it treats instance difficulty --
usually the largest source of spread in a benchmark -- as noise, and it assumes an independence the
design does not have.

On a benchmark where instance hardness has a spread of 40 and the proposed method is consistently
about 1.0 better, the same twenty instances give:

| analysis | p-value | verdict |
| --- | --- | --- |
| unpaired Welch t | 0.945 | no evidence of any effect |
| paired t (17 win / 3 loss / 0 tie) | 0.00015 | `statistically_promising` |

Same rows, opposite conclusion. So `design` is now detected from the data *before* any test runs,
and the test that design implies is the one reported. There is no second p-value on offer
afterwards.

Detection is a check, not an assumption. Pairing is used only when the identity columns
(`instance_id`, `instance`, `seed`, `replication`) line up one-to-one across the two methods; if a
key repeats within one method the pairing would be ambiguous, and an ambiguous pairing is worse than
none because it invents a design silently. Every refusal says why, and the fix is to make the
experiment solve the same instances with both methods -- not to re-run the analysis a different way.

Two traps a paired analysis brings with it are reported rather than hidden:

- **Ties.** Both methods proving the optimum on an easy instance is a tie, not evidence. When ties
  are half the benchmark or more, the report says how few instances the verdict actually rests on
  and prints the win/loss/tie counts next to the p-value.
- **A constant difference.** If every paired difference is identical there is no variation to test,
  and two real search methods almost never differ by a constant on every instance. The report says
  so and points at the likely cause: one method being computed from the other. That was exactly the
  defect in the original synthetic template.

When SciPy is present a Wilcoxon signed-rank test runs alongside the paired t-test. It is allowed
to *raise doubt* about a verdict and never to upgrade one -- a procedure that reports whichever of
two tests agrees with the hypothesis is not two tests, it is one biased test.

## Transformation Equivalence: Bilevel to KKT to MI 0-1

A reformulation that is wrong does not crash. It solves, it returns a number, and the number is
different. This is the one correctness question operations research can actually settle and a
machine-learning pipeline cannot, because the original problem has a computable answer to
compare against.

```powershell
python -m src.main run --config configs\bilevel_transformation.yaml
```

For each generated bilevel program the study establishes the true optimum with an oracle that
shares nothing with the transformation under test -- it enumerates the vertices of the joint
region and solves the follower's LP directly at each one -- and then checks each rewrite:

| step | how it is solved | what can go wrong |
| --- | --- | --- |
| `bilevel_to_kkt` | enumerate complementarity patterns (no Big-M) | sign errors, wrong rows carrying multipliers |
| `kkt_to_mi01` | HiGHS, cross-checked by binary enumeration | Big-M too small, or so large that tolerance defeats it |

Four verdicts are distinguished, because they have different causes:

- `equivalent` - values agree *and* the model's point survives an independent bilevel check.
- `model_worse_than_truth` - the optimum was cut off; a constant is too small.
- `model_better_than_truth` - the model beat the true optimum, so complementarity is not being
  enforced. A modelling error, not a tuning problem.
- `model_infeasible` - no solution where the original has one.

### Big-M Is Derived, Not Guessed

`derive_big_m_bounds` produces constants with provenance:

- **Slack bounds** come from interval arithmetic over the box, so they hold for every point in it.
- **Multiplier bounds** come from the vertices of the dual feasible set `{lambda >= 0 :
  G'lambda = -d2}`. Folding the `y` bounds into `G` makes that set unbounded in every
  coordinate, so simply maximising `lambda_i` over it returns infinity -- but its *vertices* are
  finite, and an LP with a bounded feasible region always admits an optimal dual among them.

An infinite constant is refused rather than replaced by a large number, because that
substitution is the failure this module exists to detect.

### What The Sweep Found

On the reference instances, sweeping the constants relative to the derived bounds gives:

| setting | optimum survives |
| --- | --- |
| derived x 0.1 | 0% (model infeasible) |
| derived x 0.5 | 0% (model infeasible) |
| derived x 1 | 100% |
| derived x 10 | 100% |
| uniform M = 1e3 | 100% |
| uniform M = 1e6 | 92% |

Two things worth noting. The derived bounds are close to *necessary*: halving them already
destroys the model. And the folklore "just set M large" fails -- at `M = 1e6` a binary can sit at
`5e-7`, inside the solver's integrality tolerance, while `lambda <= M z` still permits a
multiplier of 0.5. Complementarity is then not enforced at all: on the textbook instance the
model reports -21 against a true optimum of -12, with a follower who is not optimising. The two
independent MIP solves disagree, which is how it is caught.

## QUBO Encoding: The Last Hop Before A Sampler

A QUBO has no constraints and no continuous variables, so getting there costs two things that
fail differently. Lumping them together makes both undiagnosable, so the chain is measured in
two places:

```text
source MIP --(discretise)--> binary grid MIP --(penalise)--> QUBO
             |                                 |
             +-- cost of the grid              +-- correctness of the penalty
```

The binary grid model is still constrained, so it can be solved exactly by the ordinary MIP
machinery. Its optimum can only be *worse* than the source optimum, because the grid is a subset
of the feasible set -- if it ever comes out better, the projection is wrong, and that is reported
as a defect rather than as good news.

```powershell
python -m src.main run --config configs\qubo_transformation.yaml
```

Verdicts distinguish four different things:

- `equivalent` - the proved ground state decodes to the grid optimum.
- `discretisation_loss` - the optimum is not on the grid. A **measured cost**, not a bug.
- `penalty_too_small` - the ground state decodes to an infeasible point. A sampler would return
  it as the answer with no sign of trouble.
- `not_proved` - a sampler was used. Matching the optimum is not proving it.

### What The Study Found

**The derived penalty bound is rigorous but loose.** With integral data every residual is an
integer, so `P > sum_k |c_k|` is genuinely sufficient. Sweeping downwards shows the real
threshold sits around `0.003x` that bound -- roughly three hundred times smaller. Since the
penalty inflates coefficients and hardware precision is finite, that looseness is not free.

**Slack registers, not variables, are what make QUBOs expensive.** Every inequality needs its own
register:

| family | variables | constraints | QUBO bits | slack bits | ground state provable |
| --- | --- | --- | --- | --- | --- |
| knapsack | 8 | 1 | 16 | 8 | yes |
| set_cover | 8 | 6 | 14 | 6 | yes |
| multi_knapsack | 8 | 3 | 32 | 24 | no |

Three capacity constraints cost 24 bits and put the model beyond exhaustive proof entirely. Past
that point the study reports `not_proved` instead of accepting an annealer's best energy as the
optimum.

**Sampler non-convergence is not an encoding defect.** An annealer can return a bitstring whose
variables are feasible while its slack register is simply wrong, which shows up as an energy that
does not match the objective. That is the sampler failing to converge, so an encoding defect is
only ever diagnosed from a *proved* ground state.

## The Search Is A Tree Now, Not A Chain

Until this layer the controller held one `parent_workspace` that moved to the best node at the end
of each iteration. Every branch in an iteration therefore mutated the same parent, a failed node
was a dead end, and `bfts_policy_queue.md` ranked nodes for expansion *after* the search had
finished expanding. The report described a tree the search never built.

`src/core/search_policy.py` moves that ranking to where it can act. For each branch slot the policy
answers one question -- which kind of node, from which parent -- and records the candidates it
weighed:

- **draft** opens a new root from the base code. Until `search_num_drafts` roots exist the policy
  drafts, because a search with one root cannot recover from a bad start.
- **debug** copies a failed leaf and hands it to the repair backend. Bounded by
  `search_max_debug_depth`, and reached with probability `search_debug_probability`, so one broken
  branch can neither be abandoned instantly nor consume the whole budget.
- **improve** mutates the highest-priority successful node: objective value plus a
  `1 / (1 + children)` exploration bonus, so a strong node stops crowding out its untried siblings
  once it has been expanded.

Configure it in the run config:

```yaml
search_num_drafts: 2
search_max_debug_depth: 2
search_debug_probability: 0.5
search_exploration_weight: 1.0
search_seed: 0            # fixed, so a search can be replayed
```

Every decision lands in `search_policy_trace.md` with its reason and its rejected alternatives, and
`bfts_policy_queue.md` now carries `selection_basis: observed` -- the nodes that really were
expanded -- instead of a guess at what would be expanded next.

### A Failure Is A Branch, Not A Dead End

A `debug` node does not get the strategy's next idea; it gets the repair. Telling the refiner to
pursue a new hypothesis while the script is still broken is how one failure becomes two.

The evidence handed to the repair backend is a `RepairContext` built by
`src/execution/traceback_parser.py`, not the raw wall of stderr. Frames belonging to this project's
own machinery are dropped -- otherwise whatever reads them proposes fixes to the framework rather
than to the generated experiment -- absolute workspace paths are rewritten back to `experiment.py`,
the failing line is quoted with four lines of context, and every repair already tried on that
branch is listed so the same fix is not proposed twice.

Nothing is invented. A timeout or a bare non-zero exit yields no `exception_type` rather than a
guessed one, and the policy treats an unparsed failure as less actionable than a named one, because
it is.

When the backend declines to change anything, the node is closed without re-running. Executing
byte-identical code buys the same failure for the price of a timeout. The branch is marked so the
policy never offers it again.

### A Rewrite Is Accepted For What It Still Guarantees

`code_editing_backend: llm` lets a model rewrite the whole script, which is the only way to fix a
failure nobody wrote a rule for and also the only way to quietly destroy the experiment. A rewrite
that deletes the solve loop still compiles. One that writes an empty `results.csv` and prints the
summary line still satisfies the experiment contract.

So the model never writes to disk. Its output goes through `src/core/generated_code_guard.py`
first, which requires that the script parses, still writes `results.csv`, still prints
`SUMMARY_JSON:`, imports nothing that leaves the sandbox, calls no `eval`/`exec`/shell, writes
nowhere but its own workspace, and is not a gutted stub -- a "fix" that deletes the experiment
passes every other check on that list, which is why the last one exists. Imports and calls are
checked on the parse tree, because a string search for `import socket` is fooled by
`from urllib import request as r`.

A rejection is specific enough to be handed straight back as the next instruction, so the second
attempt is a correction rather than a retry. A model that cannot satisfy the checks within
`llm_repair_attempts` leaves the workspace untouched. Every prompt, every response and the original
script are saved in the node directory, so a repair that made things worse is readable and
recoverable.

None of this makes a rewrite *correct*. It makes it bounded: the ways it can still be wrong are
ways a human reading `results.csv` can notice.

The two repair paths differ in cost, and the cheaper one runs first. A node repairs itself in place
and re-runs (`max_debug_attempts`); only a failure that survives its own node is worth a whole
branch, and the debug node then starts from a fresh workspace carrying every repair already tried
along that branch.

```yaml
code_editing_backend: llm     # deterministic | llm | aider | auto | disabled
llm_repair_attempts: 2        # rejections are fed back, so this is a conversation, not a retry
```

## What The Previous Runs Learned

`research_memory/runs.jsonl` was written every run and never opened again, so each run started from
nothing: the same dead end could be explored on Monday and again on Tuesday with no record that
Monday had already tried it. `src/utils/research_memory.py` closes that loop.

A finished run records what it measured -- best metric, node counts, the statistical verdict,
verification failures, which mutation kinds preceded an improvement *over their own parent* rather
than over the run's best, and the hypotheses it tested and did not support. The next run matches
prior goals by content-word overlap and writes the matches to `prior_findings.md`, which is fed to
the idea agent and to the search strategy before either chooses anything.

The wording is deliberate. A prior run is evidence about *that run*: a hypothesis is recorded as
"was not supported in run X", never as "is false", and the brief says in as many words that a
single unsupported result is weak evidence. The strategy agent spends a spare branch slot
*retesting* what an earlier run left unsupported rather than pruning it -- and only when a slot is
free, so memory never displaces the planner's own branches. A search that treats one underpowered
comparison as settled is a search that has stopped looking.

Reading memory can never cost a run: an unparseable line is skipped, and a failure to write the
record leaves `research_memory_error.txt` beside a report that is already on disk.

## The Front Half Now Admits What It Does Not Know

The experiment loop was built so a number could not be trusted without a check behind it. The
stages *before* it had the opposite problem, and a worse one: they did not fail, they filled in.

Measured on a real run with the research goal *"transformation rules from bilevel programming to a
mixed 0-1 program, solved with a quantum algorithm"*, the old pipeline produced:

- a **research gap** about supply-chain chaos, bullwhip ratios and multi-echelon replenishment --
  the same three paragraphs for every study ever run, with the goal pasted above them;
- an **algorithm plan** proposing dynamic programming over integer weights and a value-density
  greedy with capacity-preserving swaps: a knapsack algorithm, for every research topic;
- a **model critique** whose first line was "the formulation is still a scaffold", emitted without
  reading the draft -- true by luck, and it would have said the same about a finished model;
- an **automated review** scoring `overall: 6/10` without reading the report. That constant was
  written into cross-run memory and read by the readiness assessment as if it were a measurement;
- **novelty queries** naming bilevel, KKT, MPEC, QUBO and QAOA whatever the study was about;
- a **domain classification** of `optimization` at `confidence: 0.45` from zero keyword matches --
  the vocabulary had no term for bilevel, KKT, QUBO or quantum, and no term in any language but
  English.

A confident wrong answer is worse than an absent one, because an absent one tells you where to
look. Each stage now declares one of three things, and `artifact_provenance.md` collects the
declarations so the question *"how much of this report is actually about my study?"* has an answer
that does not require recognising the boilerplate:

| source | meaning |
| --- | --- |
| `llm` | a model wrote it, from a prompt this run actually sent |
| `derived` | computed from this run's own inputs by rules you can read in the source |
| `not_generated` | nothing wrote it; the artifact carries a `NOT GENERATED` block naming what a real answer would take |

The same run now opens with:

> 5 of 8 stage(s) produced content derived from this study's own inputs; 3 declared a gap instead
> of filling one in. In this run, no stage used a language model, so nothing here was reasoned about.

That last clause is the one that matters. With `use_mock_llm: true` -- still the default -- a run
makes zero LLM calls, and every artifact that needs reasoning says so instead of supplying prose.

`derived` is not a synonym for correct. It means the output is a function of this study's inputs
rather than a constant.

### What Became Real Instead

Two stages gained a genuine check rather than an admission:

**The critic now reads the draft.** It looks for the scaffold's own text -- `F(x, y, z; c, d,
theta)`, `I: set of decision entities`, the constraint *families* listed where constraints should
be -- and reports which of them survived, which sections are missing, and whether the document
contains a single summation, inequality or quantifier anywhere. On the current scaffold it reports
6 of 6 elements untouched and no inequality in the entire "model". It needs no LLM, and it will
start saying something different the day the modeling stage produces something specific -- which
is exactly when a stale constant would have become dangerous.

**The classifier reports what it matched.** Zero hits now means `confidence: 0.0` and "this is a
fallback, not a classification", instead of a 0.45 that reads like a weak match. Confidence is the
winner's margin over the runner-up, so a goal that matches two domains equally is reported as the
genuinely ambiguous goal it is, and `matched_terms` shows the evidence. Bilevel, KKT, MPEC, QUBO,
Ising, quantum and annealing were added to the vocabulary, in English and Traditional Chinese --
a goal written in Chinese previously matched nothing at all.

## The Literature Is Read, And Says How Well

The index stored one thing about each paper that resembled metadata:
`title_guess = filename`. A paper downloaded as `1-s2.0-S0377221723004538-main.pdf` had that
string as its title everywhere downstream -- in the novelty report's "potentially related works",
in `references.bib` as a `@misc` title, in the relevance scoring. No authors, no year, no DOI, no
abstract. Nothing had read the paper, so nothing could say what it concluded.

Extraction from a PDF is guesswork, but the guesses differ enormously in quality. A DOI matched by
regex is near-certain and makes the record checkable in one click. A title from the PDF's own
metadata is usually right. A title inferred from page-one layout is a heuristic. A title taken
from the file name is not extraction at all. Collapsing those into one column is exactly how a
file name ends up in a bibliography, so `src/utils/bibliography.py` keeps them apart:

| field | sources, best first |
| --- | --- |
| `doi` | `doi_regex` |
| `title` | `pdf_metadata` → `first_page` → `filename` |
| `authors` | `pdf_metadata` → `first_page` |
| `year` | `first_page` (preferring a year next to ©/published/accepted, so page numbers and cited years lose) |
| `abstract` | `first_page` (between *Abstract* and *Keywords*/*Introduction*) |

`is_known` is false for anything that came from the file name or nowhere, and a record is
`verifiable` only with a DOI, or a title *and* a year the document actually supplied — a bare
title string identifies a download, not a publication. `literature_review.md` opens with the count:

> 12 document(s) indexed. 9 carry a DOI and 10 are verifiable against the published record; 2 have
> no title but the file name, so they identify a file rather than a paper. 8 abstract(s) were extracted.

`references.bib` follows the same rule. An identified record becomes `@article{doe2023, ...}` with
the author, year and DOI that were read. An unidentified one stays `@misc` and carries
`note = {NOT A CITATION: no metadata was read from the document...}`. No field is ever invented: a
plausible-looking guess in a `.bib` file is the kind of error that survives into a submitted paper.

### And It Now Reaches Ideation

The uploaded papers previously stopped at the index. `IdeaAgent`'s `background_text` was the
static domain-profile YAML, so a user who supplied twenty papers got ideas that had seen none of
them. A digest of the indexed corpus -- titles, authors, years, DOIs and abstracts, with
unidentified files marked as such -- is now part of that input, and `artifact_provenance.md` lists
`literature_index` among ideation's inputs so the connection is visible rather than assumed.

## Modeling Asks, Then Checks

`ModelingAgent` was the last front-half stage with no path to a language model at all. It
interpolated the research goal into a fixed scaffold -- `F(x, y, z; c, d, theta)`, sets
`I / T / S / A`, a *list of constraint families* where the constraints belong -- and returned it as
`model_draft.md`. Configuring a real model changed nothing here, because nothing here ever called
one.

It now asks. The interesting part is what happens next, because a model asked to formulate a
problem will often write *about* modelling instead: every heading present, fluent prose,
"resource/capacity constraints, precedence constraints", an objective described but never written.
That output is indistinguishable from the old scaffold in every way that matters, and it is what
this stage has to refuse.

A draft is accepted only if it would survive the critique that is about to be written about it:

- no scaffold placeholder anywhere;
- all six required sections present;
- at least one summation, inequality, quantifier or set membership actually written down;
- an objective line beginning `minimize` or `maximize` -- not `minimize or maximize`.

`CriticAgent` and `ModelingAgent` share one implementation of that check
(`src/core/model_draft_inspection.py`), so they cannot drift into a state where modeling accepts a
draft the critique then calls a scaffold. A rejected draft's failures go back as the next
instruction rather than to disk. If no attempt passes, the scaffold is returned with the attempts
recorded in it, so a reader can see the model was asked and what it failed to produce.

A formulation nobody can implement cannot be falsified, and an unfalsifiable model is not a
research contribution — which is why "has a Constraints section" is not the test and "has a
constraint in it" is.

The literature digest reaches this stage too, alongside the domain profile's model families and any
supplied starter template.

## Turning The Real Models On

Every front-half stage now has a model path, so `use_mock_llm: false` is the switch that makes
this system do research rather than demonstrate a pipeline. Three things had to be fixed first,
and all three made the old behaviour look better than it was.

**A failed call was never recorded.** `tracker.record` ran only on success, so a run where six of
eight calls failed produced an `llm_interactions.json` showing two calls. The artifact whose whole
job is to say what the models did was silent about everything that went wrong. Failures are now
recorded per attempt, with the stage that asked, and `llm_interactions.md` opens with the count:

> 11 model call(s) succeeded and 3 failed (rate_limit x2, timeout x1). No prices were configured,
> so the spend is unknown rather than zero. The following stage(s) produced nothing as a result:
> automated_review.

**There was no retry.** A single 429 killed a stage, which then degraded to `not_generated` --
indistinguishable in the report from a stage that had no model configured at all. Transient
failures now retry with exponential backoff, honouring `Retry-After` when the provider sends one.

**A permanent failure was treated like weather.** This was the worst of the three. A mistyped key
degraded all six model-backed stages in turn, and the run finished in ten minutes with a report
saying no stage had used a language model -- true, and silent about the one thing worth knowing.
Failures are now classified by what a caller should do about them:

| kind | permanent? | meaning |
| --- | --- | --- |
| `auth`, `not_found`, `bad_request`, `budget` | yes | a configuration problem; no retry can fix it |
| `rate_limit`, `server`, `timeout`, `network`, `empty`, `malformed_json` | no | retry, then degrade honestly |

A permanent error raises, no agent swallows it, and one probe call before the pipeline starts means
a bad key costs a second rather than a whole run:

```
LLM preflight failed and the cause is permanent, so the run was stopped before doing any work:
auth: HTTP 401: Incorrect API key provided -- The provider rejected the credentials. Check the API
key in your environment (OPENAI_API_KEY / GEMINI_API_KEY) and that it has access to this model.
```

Two smaller things came with it. The OpenAI path used to let `HTTPError` escape unread, so a 404
surfaced as `HTTP Error 404: Not Found` while the body -- which says *which* thing is wrong -- was
discarded; every provider path now reads it. And a stage that degrades carries the reason with it,
so `automated_review.md` says *the review call failed (rate_limit: HTTP 429...)* rather than
implying no model was configured.

```yaml
use_mock_llm: false
llm_provider: openai          # openai | gemini | ollama
model_name: gpt-4o-mini
llm_max_attempts: 3
llm_backoff_seconds: 2.0
llm_timeout_seconds: 90
llm_preflight: true           # false to run degraded on purpose
max_llm_calls: 60             # a search loop can otherwise grow its own bill
max_llm_cost_usd: 2.00        # only enforceable if you set the prices below
llm_price_per_1k_prompt_tokens: 0.00015
llm_price_per_1k_response_tokens: 0.0006
```

There is no built-in price table. A stale price printed as a dollar figure is exactly the kind of
confident wrong number this project keeps removing, so cost is reported as *unknown* until you
supply your provider's current prices — and the call cap works either way.

## A Mutation That Changes Nothing Is Not A Branch

`code.replace(source, target)` returns the same string when `source` is absent. The mutation layer
was a list of exact replacements -- `PROPOSED_EFFECT_LOW = 0.5`, `NOISE_SCALE = 1.0`,
`REPLICATES = 10` -- and those names exist only in `generic_or_template.py`. On the binary-program
scaffold, the bilevel scaffold, the QUBO scaffold, or any template a user writes, **every
replacement missed**. The only difference between parent and child was the docstring header the
refiner inserts.

So the node was a byte-identical copy of its parent. The search then ran it, scored it, ranked it
on the best-first frontier, and reported it as a branch that had been explored. Nothing noticed,
because the refiner returned a bare string and no caller compared it to what it was given.

Two halves to the fix.

**The no-op is now visible.** The refiner returns a `MutationOutcome` saying which replacement keys
were present, which were absent, and whether the behaviour changed at all -- comparing parent and
child with comments, blank lines and the header stripped, since a variant that differs only by its
header differs only in how it describes itself. `mutation_trace.md` opens with the count:

> 3 of 6 mutation(s) actually changed the experiment. A mutation that changed nothing produced a
> node identical to its parent, which re-measures the parent rather than exploring anything.

**A model can apply the intent instead of the strings.** With `use_mock_llm: false` the refiner
asks for the mutation's *description* to be applied to the whole script -- which works on templates
the replacement keys know nothing about -- listing the reference edits as an example rather than as
the instruction. That rewrite goes through the same guard as a repair rewrite (parses, contract
preserved, no sandbox escape, not a gutted stub) plus the check above: a variant identical to its
parent is rejected and the reason fed back. A rejected rewrite is never shipped; the string path
runs instead.

The mutation prompt carries one instruction the string path never needed: *do not make the proposed
method look better by weakening the baseline or the verification.* A mutation that improves the
result by measuring less is the one kind of "successful" mutation this system must not accept.

### The Search Reads Its Own History Now

`journal`, `iteration` and `branch_index` were parameters `MutationAgent` ignored, so the same kind
came back however it had turned out before. A kind is now *spent* when every node carrying it
failed, or when the refiner reported it changed nothing -- the second being the more useful signal,
since it means the kind is inert on this template rather than that the idea was bad. When a spent
kind is the only one the domain profile offers it is still used, with a note saying so, because
leaving the slot with no mutation at all is worse.

## Paper-Inspired Automation Pattern

The implementation follows a conservative version of the workflow described in
`Towards end-to-end automation of AI research`:

- Ideation remains structured and inspectable through `idea_archive.json` and `selected_idea.json`.
- Experimentation uses a workspace tree: each node has its own `experiment.py`, `plot.py`, prompt metadata, logs, results, and figures.
- The best successful checkpoint is promoted to `best_experiment.py`, `best_plot.py`, `best_prompt.json`, and `best_notes.md`.
- Failed experiment nodes are not ignored. The system records `repair_trace.md` and retries deterministic repairs when possible.
- A failed node can be repaired by a guarded whole-file LLM rewrite, or by Aider, with edits
  restricted to the node workspace and checked before they are written.
- Plot execution is separate from experiment execution, then summarized in `plot_aggregation.md`.
- Tree-search decisions are made *in the loop* and written to `search_policy_trace.md`;
  `bfts_frontier.md` and `bfts_policy_queue.md` report what the search actually expanded.
- What earlier runs measured is read back into `prior_findings.md` before any planning agent runs.
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
- `search_policy_trace.json`
- `search_policy_trace.md`
- `prior_findings.json`
- `prior_findings.md`
- `artifact_provenance.json`
- `artifact_provenance.md`
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
- `equivalence_report.md` (transformation runs)
- `qubo_report.md` (QUBO runs)
- `instances/*.json` (the exact instance data, so any result can be re-solved independently)
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

`statistical_evidence.md` summarizes method-level means and confidence intervals, detects whether
the two compared methods solved the same instances, and runs the test that design implies -- a
paired t-test with per-instance win/loss/tie counts, or Welch's unpaired t-test when pairing was
refused, with the refusal explained. SciPy is used when available; otherwise it falls back to a
normal approximation and says so in `test_used`. A non-significant result vetoes the optimistic
evidence labels. The result still describes one instance set, and generalising from it is a
judgement no test can make for you.

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
It combines node value, exploration pressure, and failure penalties. Its `selection_basis`
field says whether the marked nodes were really expanded (`observed`) or are a ranking of what
would be expanded next (`predicted`), so the two are never confused.

`search_policy_trace.md` is the decision log: one entry per branch slot, giving the kind
chosen (draft, debug or improve), the parent, the reason, and every candidate weighed with its
priority. This is what the search did, not a reconstruction of it.

`prior_findings.md` is what earlier runs of this project measured on a similar goal, written
before ideation so the planning agents can see it. It reports prior negatives as unsupported
in that run rather than as settled, because that is all a single run establishes.

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

- Without a model, mutation is still exact string replacement, so on any template but
  `generic_or_template.py` it is inert. It now says so instead of producing a copy of the
  parent, which makes the limitation visible rather than removing it.
- `use_mock_llm: true` is still the default, so out of the box the system makes zero LLM calls
  and its model-backed stages have nothing to reason with. They say so instead of inventing
  content, which makes the limitation visible rather than removing it.
- The cost figure is local token estimates (characters / 4) times prices you configured. It is
  not an invoice, and it will drift from your provider's billing.
- Bibliographic extraction is a heuristic read of page one. It will misread unusual layouts,
  scanned pages with no text layer, and non-English papers; `title_source` and `verifiable`
  say which records to distrust, but nothing checks the extraction against a real database.
- The literature reaches ideation and modeling as a digest. No stage reads past the first six
  pages of any paper, so a method buried in section 4 is invisible.
- The model-draft inspection checks that a formulation is *specific*, never that it is
  *correct*. A draft with real symbols, real inequalities and the wrong constraints passes
  every check here; only a human can catch that.
- The LLM repair backend is only as good as the model behind it. The guard bounds what a bad
  rewrite can do; it cannot make a weak model competent.
- The search policy decides between draft, debug and improve, but the *value* it ranks on is
  the node's own metric. There is no learned value estimate, so a branch that is one bad step
  away from a good region looks the same as one that is exhausted.
- The paired analysis compares two methods. A benchmark with several methods needs a correction
  for multiple comparisons, which is not implemented; treat per-pair p-values accordingly.
- No figure is ever looked at; a blank or non-converged plot is reported as success.
- Transformation equivalence is checked on small generated instances only, and `equivalent` there
  is evidence, not a proof of correctness in general.
- The QUBO step is verified on pure 0-1 sources. A bilevel MI 0-1 model has enough continuous
  variables that its expansion runs past forty bits, so no ground state can be proved there yet.
- The annealer is a stand-in, not a quantum device. Nothing here measures what real hardware
  noise, embedding overhead or finite coefficient precision would do to these encodings.
- Cross-run memory matches prior runs by content-word overlap of the research goal. That is
  transparent and checkable, but it will miss a related run phrased in different vocabulary.

## Limitations

- The system can scaffold and run initial computational research, but it cannot replace mathematical proof or expert review.
- Literature extraction is metadata-light; citation details must be verified by a human.
- The default experiment scaffold is intentionally generic.
- Strong research claims require domain-specific validation, statistical checks, and advisor/reviewer inspection.
