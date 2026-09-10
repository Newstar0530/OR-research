from __future__ import annotations

import shutil
import json
from pathlib import Path

from src.agents.code_patch_agent import CodePatchAgent
from src.agents.ablation_agent import AblationAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.experiment_refiner_agent import ExperimentRefinerAgent
from src.agents.mutation_agent import MutationAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.strategy_agent import StrategyAgent
from src.agent_system.state import ResearchState
from src.agent_system.patch import SafePatchApplier
from src.core.budget import ResearchBudget
from src.core.decision_engine import DecisionEngine
from src.core.experiment_contract import validate_experiment_contract
from src.core.experiment_workspace import ExperimentWorkspaceManager
from src.core.research_journal import ResearchJournal
from src.core.research_node import ResearchNode
from src.core.search_policy import SearchPolicy, SearchPolicyConfig
from src.core.workspace_debugger import repair_workspace_after_failure
from src.execution.sandbox_runner import SandboxRunner
from src.execution.traceback_parser import build_repair_context, parse_traceback
from src.llm_client import LLMClient
from src.schemas import AlgorithmPlan, ResearchIdea
from src.utils.bfts_policy import build_bfts_policy_queue
from src.utils.plot_aggregator import NodePlotRecord, aggregate_node_plots


class ExperimentSearchController:
    def __init__(
        self,
        llm: LLMClient,
        runner: SandboxRunner,
        budget: ResearchBudget,
        decision_engine: DecisionEngine,
        domain_profile: str = "optimization",
        code_editing_backend: str = "deterministic",
        aider_command: str = "aider",
        aider_model: str | None = None,
        aider_timeout_seconds: int = 120,
        max_repair_attempts: int = 1,
        search_policy: SearchPolicyConfig | None = None,
        llm_repair_attempts: int = 2,
    ) -> None:
        self.llm = llm
        self.runner = runner
        self.budget = budget
        self.decision_engine = decision_engine
        self.domain_profile = domain_profile
        self.code_editing_backend = code_editing_backend
        self.aider_command = aider_command
        self.aider_model = aider_model
        self.aider_timeout_seconds = aider_timeout_seconds
        self.max_repair_attempts = max(0, max_repair_attempts)
        self.search_policy_config = (search_policy or SearchPolicyConfig()).validated()
        self.llm_repair_attempts = llm_repair_attempts
        self.planner = PlannerAgent(llm)
        self.strategy = StrategyAgent(llm)
        self.mutation_agent = MutationAgent(llm)
        self.code_patch_agent = CodePatchAgent(llm)
        self.refiner = ExperimentRefinerAgent(llm)
        self.ablation = AblationAgent(llm)
        self.evidence = EvidenceAgent(llm)

    def run(
        self,
        run_dir: Path,
        idea: ResearchIdea,
        algorithm: AlgorithmPlan,
        base_code: str,
        research_state: ResearchState | None = None,
        agent_runtime=None,
        prior_findings: str = "",
    ) -> ResearchJournal:
        journal = ResearchJournal()
        search_dir = run_dir / "autonomous_search"
        search_dir.mkdir(parents=True, exist_ok=True)
        workspace_manager = ExperimentWorkspaceManager(run_dir)
        root_workspace = workspace_manager.initialize_root(base_code, idea, algorithm)
        policy = SearchPolicy(self.search_policy_config, maximize=self.decision_engine.maximize)
        initial_branches = self.planner.run(
            idea,
            self.budget.max_branches,
            directives=research_state.search_directives if research_state else None,
        )
        strategies = []
        mutations = []
        patches = []
        repair_attempts = []
        plot_records: list[NodePlotRecord] = []
        agent_feedback_rounds = []
        for iteration in range(self.budget.max_iterations):
            if not self.budget.allow_iteration(iteration):
                break
            strategy = self.strategy.run(
                journal,
                iteration,
                initial_branches,
                self.budget.max_branches,
                self.budget.patience,
                self.budget.min_improvement,
                directives=research_state.search_directives if research_state else None,
                prior_findings=prior_findings,
            )
            strategies.append(strategy)
            if strategy.should_stop:
                break
            reserved_parents: set[str] = set()
            for branch_index, branch_plan in enumerate(strategy.branch_plans):
                if not self.budget.allow_branch(branch_index):
                    continue
                # The parent is chosen here, per slot, rather than being whatever
                # node happened to win the previous iteration. That is the whole
                # difference between a search tree and a chain.
                decision = policy.decide(
                    journal,
                    iteration,
                    branch_index,
                    reserved_parent_ids=reserved_parents,
                    root_work_dir=root_workspace,
                )
                if decision.parent_id:
                    reserved_parents.add(decision.parent_id)
                parent_node = journal.node_by_id(decision.parent_id) if decision.parent_id else None
                source_workspace = (
                    Path(decision.parent_work_dir) if decision.parent_work_dir else root_workspace
                )
                plan = branch_plan if decision.kind != "debug" else self._debug_plan(parent_node, branch_plan)
                node = ResearchNode(
                    parent_id=decision.parent_id,
                    iteration=iteration,
                    branch_index=branch_index,
                    plan=plan,
                    kind=decision.kind,
                    depth=decision.depth,
                    debug_depth=decision.debug_depth,
                    metric_name=self.decision_engine.primary_metric,
                    maximize=self.decision_engine.maximize,
                    metadata={
                        "algorithm_metrics": algorithm.evaluation_metrics,
                        "search_decision": decision.model_dump(),
                    },
                )
                node_dir = search_dir / f"iter_{iteration:02d}_branch_{branch_index:02d}_{decision.kind}_{node.id}"
                node_dir = workspace_manager.create_node_workspace(source_workspace, node_dir, plan)
                node.metadata["source_workspace"] = str(source_workspace)
                code_path = node_dir / "experiment.py"
                node.code_path = str(code_path)
                node.work_dir = str(node_dir)
                attempted_repairs: list[str] = self._previous_repair_reasons(journal, parent_node)

                if decision.kind == "debug":
                    inherited = self._repair_inherited_failure(node, parent_node, node_dir, attempted_repairs)
                    repair_attempts.append(
                        {
                            "node_id": node.id,
                            "repair_index": -1,
                            "stage": "inherited",
                            "inherited_from": decision.parent_id,
                            **inherited.model_dump(),
                        }
                    )
                    node.metadata["inherited_repair"] = inherited.model_dump()
                    if not inherited.applied:
                        # The workspace is byte-identical to the parent's. Running it
                        # would reproduce the same failure and spend a timeout proving
                        # something already known, so the node is closed here and
                        # marked so the policy stops offering this branch for repair.
                        node.status = "contract_failed"
                        node.analysis = (
                            f"No repair was applied to the inherited failure, so this branch was"
                            f" closed without re-running. Backend `{inherited.backend}` said:"
                            f" {inherited.reason}"
                        )
                        node.failure_summary = f"repair declined: {inherited.reason}"
                        node.metadata["repair_exhausted"] = True
                        journal.append(node)
                        continue
                    attempted_repairs.append(inherited.reason)
                else:
                    mutation = self.mutation_agent.run(journal, iteration, branch_index, plan, self.domain_profile)
                    mutations.append({"node_id": node.id, "iteration": iteration, "branch_index": branch_index, **mutation.model_dump()})
                    node.metadata["mutation"] = mutation.model_dump()
                    parent_code = code_path.read_text(encoding="utf-8")
                    outcome = self.refiner.run(parent_code, plan, iteration, branch_index, mutation)
                    code = outcome.code
                    code_path.write_text(code, encoding="utf-8")
                    node.metadata["mutation_outcome"] = outcome.model_dump()
                    mutations[-1]["outcome"] = outcome.model_dump()
                    if not outcome.is_effective:
                        # The variant is its parent. Running it re-measures the
                        # parent, which is worth knowing but is not exploration,
                        # so it must not be reported as an explored branch.
                        node.analysis = ""
                        node.metadata["mutation_had_no_effect"] = True
                    proposal = self.code_patch_agent.run(code, plan, mutation)
                    if proposal:
                        patch_result = SafePatchApplier(node_dir).apply(proposal)
                        patches.append(
                            {
                                "node_id": node.id,
                                "iteration": iteration,
                                "branch_index": branch_index,
                                "proposal": proposal.model_dump(),
                                "result": patch_result.model_dump(),
                            }
                        )
                        node.metadata["patch"] = {"proposal": proposal.model_dump(), "result": patch_result.model_dump()}
                execution = self.runner.run_python(code_path, node_dir)
                contract = validate_experiment_contract(node_dir)
                status, metric_value, analysis = self.decision_engine.evaluate(node_dir / "results.csv", execution, contract)
                if status != "success":
                    for repair_index in range(self.max_repair_attempts):
                        repair = repair_workspace_after_failure(
                            node_dir,
                            execution.stderr,
                            stdout=execution.stdout,
                            attempt=repair_index,
                            backend=self.code_editing_backend,
                            aider_command=self.aider_command,
                            aider_model=self.aider_model,
                            timeout_seconds=self.aider_timeout_seconds,
                            previous_attempts=attempted_repairs,
                            llm=self.llm,
                            llm_attempts=self.llm_repair_attempts,
                        )
                        repair_attempts.append(
                            {"node_id": node.id, "repair_index": repair_index, "stage": "in_node", **repair.model_dump()}
                        )
                        if not repair.applied:
                            break
                        attempted_repairs.append(repair.reason)
                        execution = self.runner.run_python(code_path, node_dir)
                        contract = validate_experiment_contract(node_dir)
                        status, metric_value, analysis = self.decision_engine.evaluate(node_dir / "results.csv", execution, contract)
                        node.metadata["repair_retry"] = repair.model_dump()
                        if status == "success":
                            break
                comparison = self.decision_engine.method_comparison(node_dir / "results.csv")
                node.execution = execution
                node.status = status  # type: ignore[assignment]
                node.metric_value = metric_value
                node.analysis = analysis
                if status != "success":
                    # Parse the failure now, while the workspace still exists. The
                    # policy reads these fields to decide whether the failure is
                    # actionable enough to be worth a debug slot.
                    self._record_failure(node, node_dir, execution)
                experiment_log = node_dir / "subprocess_last.log"
                if experiment_log.exists():
                    shutil.copyfile(experiment_log, node_dir / "experiment_last.log")
                if comparison:
                    node.metadata["method_comparison"] = comparison
                    node.analysis = analysis + f" Method comparison: {comparison.get('reason')}"
                node.artifacts = {
                    "results_csv": str(node_dir / "results.csv"),
                    "contract_report": str(node_dir / "experiment_contract.md"),
                    "experiment_log": str(node_dir / "experiment_last.log"),
                }
                (node_dir / "experiment_contract.md").write_text(contract.to_markdown(), encoding="utf-8")
                plot_records.append(self._run_node_plot(node.id, node_dir))
                journal.append(node)
            if research_state is not None:
                self._update_research_state_iteration(research_state, journal, iteration)
                self._save_state_snapshot(run_dir, research_state, iteration)
                if agent_runtime is not None and iteration < self.budget.max_iterations - 1:
                    before = agent_runtime.message_count()
                    agent_runtime.run_round(research_state)
                    after = agent_runtime.message_count()
                    agent_feedback_rounds.append(
                        {
                            "iteration": iteration,
                            "messages_before": before,
                            "messages_after": after,
                            "search_directives": list(research_state.search_directives),
                        }
                    )
        journal.save_json(run_dir / "autonomous_journal.json")
        (run_dir / "autonomous_journal.md").write_text(journal.to_markdown(), encoding="utf-8")
        (run_dir / "strategy_trace.json").write_text(
            json.dumps([strategy.model_dump() for strategy in strategies], indent=2),
            encoding="utf-8",
        )
        (run_dir / "strategy_trace.md").write_text(
            "# Strategy Trace\n\n" + "\n".join(strategy.to_markdown() for strategy in strategies),
            encoding="utf-8",
        )
        (run_dir / "mutation_trace.json").write_text(json.dumps(mutations, indent=2), encoding="utf-8")
        (run_dir / "mutation_trace.md").write_text(self._mutation_markdown(mutations), encoding="utf-8")
        (run_dir / "patch_trace.json").write_text(json.dumps(patches, indent=2), encoding="utf-8")
        (run_dir / "patch_trace.md").write_text(self._patch_markdown(patches), encoding="utf-8")
        (run_dir / "agent_feedback_rounds.json").write_text(json.dumps(agent_feedback_rounds, indent=2), encoding="utf-8")
        (run_dir / "agent_feedback_rounds.md").write_text(self._feedback_rounds_markdown(agent_feedback_rounds), encoding="utf-8")
        (run_dir / "workspace_lineage.json").write_text(json.dumps(self._workspace_lineage(journal), indent=2), encoding="utf-8")
        (run_dir / "workspace_lineage.md").write_text(self._workspace_lineage_markdown(journal), encoding="utf-8")
        (run_dir / "bfts_frontier.json").write_text(json.dumps(self._bfts_frontier(journal), indent=2), encoding="utf-8")
        (run_dir / "bfts_frontier.md").write_text(self._bfts_frontier_markdown(journal), encoding="utf-8")
        (run_dir / "search_policy_trace.json").write_text(
            json.dumps([decision.model_dump() for decision in policy.decisions], indent=2),
            encoding="utf-8",
        )
        (run_dir / "search_policy_trace.md").write_text(policy.trace_markdown(), encoding="utf-8")
        (run_dir / "repair_trace.json").write_text(json.dumps(repair_attempts, indent=2), encoding="utf-8")
        (run_dir / "repair_trace.md").write_text(self._repair_trace_markdown(repair_attempts), encoding="utf-8")
        aggregate_node_plots(plot_records, run_dir)
        build_bfts_policy_queue(
            journal,
            run_dir,
            "maximize" if self.decision_engine.maximize else "minimize",
            expanded_parent_ids=[d.parent_id for d in policy.decisions if d.parent_id],
        )
        (run_dir / "ablation_plan.md").write_text(self.ablation.run(journal), encoding="utf-8")
        (run_dir / "evidence_synthesis.md").write_text(self.evidence.run(journal), encoding="utf-8")
        if research_state is not None:
            self._update_research_state(research_state, journal)
        self._promote_best_outputs(journal, run_dir)
        return journal

    # -- debug-node plumbing -----------------------------------------------

    @staticmethod
    def _debug_plan(parent_node: ResearchNode | None, branch_plan: str) -> str:
        """A debug node's plan is the repair, not the strategy's next idea.

        Handing it the strategy's branch plan would tell the refiner to change
        the experiment while it is still broken, which is how a failure turns
        into two failures.
        """

        if parent_node is None:
            return branch_plan
        failure = parent_node.failure_summary or parent_node.exception_type or "an unrecorded failure"
        return (
            f"Repair node `{parent_node.id}`, which failed with {failure}."
            " Change only what is needed to make the script run; do not pursue"
            f" the pending idea in the meantime ({branch_plan})."
        )

    def _repair_inherited_failure(
        self,
        node: ResearchNode,
        parent_node: ResearchNode | None,
        node_dir: Path,
        previous_attempts: list[str],
    ):
        """Repair the failure this node inherited from its parent.

        The evidence is the parent's stderr, read against this node's own copy of
        the workspace, and every repair already tried along this debug chain is
        passed in so the backend is not invited to re-propose a fix that has
        already been shown not to work.
        """

        execution = parent_node.execution if parent_node else None
        stderr = execution.stderr if execution else ""
        stdout = execution.stdout if execution else ""
        context = build_repair_context(
            stderr,
            node_dir,
            script_name="experiment.py",
            stdout=stdout,
            attempt=node.debug_depth,
            previous_attempts=previous_attempts,
        )
        node.exception_type = context.exception_type
        node.exception_message = context.exception_message
        node.failing_line = context.failing_line
        node.metadata["inherited_failure"] = {
            "parent_id": parent_node.id if parent_node else None,
            "exception_type": context.exception_type,
            "exception_message": context.exception_message,
            "failing_line": context.failing_line,
            "actionable": context.is_actionable,
        }
        return repair_workspace_after_failure(
            node_dir,
            stderr,
            stdout=stdout,
            attempt=node.debug_depth,
            backend=self.code_editing_backend,
            aider_command=self.aider_command,
            aider_model=self.aider_model,
            timeout_seconds=self.aider_timeout_seconds,
            context=context,
            llm=self.llm,
            llm_attempts=self.llm_repair_attempts,
        )

    @staticmethod
    def _previous_repair_reasons(journal: ResearchJournal, node: ResearchNode | None) -> list[str]:
        """Every repair already attempted on this branch, root-most first."""

        reasons: list[str] = []
        seen: set[str] = set()
        current = node
        while current is not None and current.id not in seen:
            seen.add(current.id)
            for key in ("inherited_repair", "repair_retry"):
                entry = current.metadata.get(key)
                if isinstance(entry, dict) and entry.get("reason"):
                    reasons.append(str(entry["reason"]))
            current = journal.node_by_id(current.parent_id) if current.parent_id else None
        reasons.reverse()
        return reasons

    @staticmethod
    def _record_failure(node: ResearchNode, node_dir: Path, execution) -> None:
        """Store the parsed failure on the node.

        A node that failed without a parseable traceback -- a timeout, a bare
        non-zero exit -- keeps `exception_type` as None rather than being given
        an invented one. The policy treats that as less actionable, which is
        exactly what it is.
        """

        parsed = parse_traceback(getattr(execution, "stderr", "") or "", node_dir, "experiment.py")
        node.exception_type = parsed.exception_type
        node.exception_message = parsed.exception_message
        frame = parsed.failing_frame
        node.failing_line = frame.lineno if frame else None
        node.failure_summary = parsed.summary()

    def _promote_best_outputs(self, journal: ResearchJournal, run_dir: Path) -> None:
        best = journal.best_node()
        if best is None or not best.work_dir:
            return
        best_dir = Path(best.work_dir)
        for name in [
            "results.csv",
            "subprocess_last.log",
            "experiment_last.log",
            "verification_report.md",
            "solver_backend_report.json",
            "experiment.py",
            "prompt.json",
            "notes.md",
            "plot.py",
        ]:
            src = best_dir / name
            if src.exists():
                promoted_as_is = {
                    "results.csv",
                    "subprocess_last.log",
                    "experiment_last.log",
                    "verification_report.md",
                    "solver_backend_report.json",
                }
                target_name = name if name in promoted_as_is else f"best_{name}"
                shutil.copyfile(src, run_dir / target_name)
        src_figures = best_dir / "figures"
        dst_figures = run_dir / "figures"
        if src_figures.exists():
            dst_figures.mkdir(exist_ok=True)
            for item in src_figures.glob("*"):
                if item.is_file():
                    shutil.copyfile(item, dst_figures / item.name)

    def _update_research_state(self, state: ResearchState, journal: ResearchJournal) -> None:
        best = journal.best_node()
        state.artifacts.update(
            {
                "autonomous_journal": "autonomous_journal.md",
                "strategy_trace": "strategy_trace.md",
                "mutation_trace": "mutation_trace.md",
                "patch_trace": "patch_trace.md",
                "evidence_synthesis": "evidence_synthesis.md",
            }
        )
        state.metadata["autonomous_nodes"] = len(journal.nodes)
        if best:
            state.evidence.append(
                f"Best autonomous node `{best.id}` achieved {best.metric_name}={best.metric_value}; status={best.status}."
            )
            comparison = best.metadata.get("method_comparison")
            if isinstance(comparison, dict):
                state.metadata["best_method_comparison"] = comparison
                state.evidence.append(
                    f"Method comparison improvement={comparison.get('improvement')} supports_hypothesis={comparison.get('supports_hypothesis')}."
                )

    def _update_research_state_iteration(self, state: ResearchState, journal: ResearchJournal, iteration: int) -> None:
        nodes = journal.nodes_for_iteration(iteration)
        best = None
        for node in nodes:
            if node.better_than(best):
                best = node
        summary = {
            "iteration": iteration,
            "nodes": len(nodes),
            "best_node": best.id if best else None,
            "best_metric": best.metric_value if best else None,
            "best_status": best.status if best else None,
        }
        if best and isinstance(best.metadata.get("method_comparison"), dict):
            summary["method_comparison"] = best.metadata["method_comparison"]
        state.metadata["latest_iteration_summary"] = summary
        state.evidence.append(
            f"Iteration {iteration} completed with {len(nodes)} node(s); best={summary['best_node']} metric={summary['best_metric']}."
        )

    @staticmethod
    def _save_state_snapshot(run_dir: Path, state: ResearchState, iteration: int) -> None:
        snapshot_dir = run_dir / "state_snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        (snapshot_dir / f"after_iteration_{iteration:02d}.json").write_text(
            json.dumps(state.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _mutation_markdown(mutations: list[dict]) -> str:
        lines = ["# Mutation Trace", ""]
        if not mutations:
            lines.append("No mutations were applied.")
            return "\n".join(lines) + "\n"
        # The number worth reading first: a node whose mutation changed nothing
        # is a copy of its parent, however it was scored.
        effective = sum(
            1
            for item in mutations
            if isinstance(item.get("outcome"), dict) and item["outcome"].get("applied")
        )
        lines.append(
            f"{effective} of {len(mutations)} mutation(s) actually changed the experiment. A "
            "mutation that changed nothing produced a node identical to its parent, which "
            "re-measures the parent rather than exploring anything."
        )
        lines.append("")
        for mutation in mutations:
            lines.append(f"## Node {mutation.get('node_id')}")
            lines.append(f"- iteration: {mutation.get('iteration')}")
            lines.append(f"- branch: {mutation.get('branch_index')}")
            lines.append(f"- kind: {mutation.get('kind')}")
            lines.append(f"- description: {mutation.get('description')}")
            outcome = mutation.get("outcome")
            if isinstance(outcome, dict):
                lines.append(f"- method: {outcome.get('method')}")
                lines.append(f"- changed the script: {outcome.get('applied')}")
                unmatched = outcome.get("unmatched_replacements") or []
                if unmatched:
                    lines.append(
                        "- replacements that matched nothing in this template: "
                        + ", ".join(f"`{item}`" for item in unmatched)
                    )
                for note in outcome.get("notes") or []:
                    lines.append(f"- note: {note}")
            replacements = mutation.get("replacements", {})
            if replacements:
                lines.append("- replacements:")
                for source, target in replacements.items():
                    lines.append(f"  - `{source}` -> `{target}`")
            lines.append("")
        return "\n".join(lines)

    def _run_node_plot(self, node_id: str, node_dir: Path) -> NodePlotRecord:
        plot_path = node_dir / "plot.py"
        if not plot_path.exists():
            return NodePlotRecord(node_id=node_id, work_dir=str(node_dir), plot_status="missing_plot_py")
        # A separate log name: reusing subprocess_last.log here would overwrite the
        # experiment's own stdout, which the experiment contract needs to verify.
        execution = self.runner.run_python(plot_path, node_dir, log_name="plot_subprocess.log")
        figures = []
        figures_dir = node_dir / "figures"
        if figures_dir.exists():
            figures = [str(path) for path in figures_dir.glob("*") if path.is_file()]
        log_path = node_dir / "plot_last.log"
        log_path.write_text(f"STDOUT\n{execution.stdout}\n\nSTDERR\n{execution.stderr}\n", encoding="utf-8")
        return NodePlotRecord(
            node_id=node_id,
            work_dir=str(node_dir),
            plot_status=execution.status,
            plot_log=str(log_path),
            figures=figures,
        )

    @staticmethod
    def _repair_trace_markdown(repairs: list[dict]) -> str:
        lines = ["# Repair Trace", ""]
        if not repairs:
            lines.append("No failed-node repair attempts were needed.")
            return "\n".join(lines) + "\n"
        for item in repairs:
            lines.append(f"## Node {item.get('node_id')}")
            lines.append(f"- attempted: {item.get('attempted')}")
            lines.append(f"- applied: {item.get('applied')}")
            lines.append(f"- reason: {item.get('reason')}")
            notes = item.get("notes") or []
            if notes:
                lines.append("- notes:")
                lines.extend(f"  - {note}" for note in notes)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _patch_markdown(patches: list[dict]) -> str:
        lines = ["# Patch Trace", ""]
        if not patches:
            lines.append("No safe code patches were proposed or applied.")
            return "\n".join(lines) + "\n"
        for patch in patches:
            result = patch.get("result", {})
            proposal = patch.get("proposal", {})
            lines.append(f"## Node {patch.get('node_id')}")
            lines.append(f"- iteration: {patch.get('iteration')}")
            lines.append(f"- branch: {patch.get('branch_index')}")
            lines.append(f"- applied: {result.get('applied')}")
            lines.append(f"- reason: {result.get('reason')}")
            lines.append(f"- rationale: {proposal.get('rationale')}")
            operations = proposal.get("operations", [])
            if operations:
                lines.append("- operations:")
                for operation in operations:
                    lines.append(f"  - `{operation.get('find')}` -> `{operation.get('replace')}`")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _feedback_rounds_markdown(rounds: list[dict]) -> str:
        lines = ["# Agent Feedback Rounds", ""]
        if not rounds:
            lines.append("No in-loop agent feedback rounds were executed.")
            return "\n".join(lines) + "\n"
        for item in rounds:
            lines.append(f"## After Iteration {item.get('iteration')}")
            lines.append(f"- messages_before: {item.get('messages_before')}")
            lines.append(f"- messages_after: {item.get('messages_after')}")
            lines.append("- search_directives:")
            for directive in item.get("search_directives", [])[-5:]:
                lines.append(f"  - {directive}")
            lines.append("")
        return "\n".join(lines)

    def _workspace_lineage(self, journal: ResearchJournal) -> list[dict]:
        return [
            {
                "node_id": node.id,
                "parent_id": node.parent_id,
                "iteration": node.iteration,
                "branch_index": node.branch_index,
                "source_workspace": node.metadata.get("source_workspace"),
                "work_dir": node.work_dir,
                "status": node.status,
                "metric_value": node.metric_value,
            }
            for node in journal.nodes
        ]

    def _workspace_lineage_markdown(self, journal: ResearchJournal) -> str:
        lines = ["# Workspace Lineage", ""]
        if not journal.nodes:
            return "# Workspace Lineage\n\nNo workspace nodes were executed.\n"
        for item in self._workspace_lineage(journal):
            lines.append(f"## Node {item['node_id']}")
            lines.append(f"- parent: {item['parent_id'] or 'none'}")
            lines.append(f"- source_workspace: `{item['source_workspace']}`")
            lines.append(f"- work_dir: `{item['work_dir']}`")
            lines.append(f"- status: {item['status']}")
            lines.append(f"- metric_value: {item['metric_value'] if item['metric_value'] is not None else 'n/a'}")
            lines.append("")
        return "\n".join(lines)

    def _bfts_frontier(self, journal: ResearchJournal) -> list[dict]:
        scored = []
        for node in journal.nodes:
            if node.status != "success" or node.metric_value is None:
                score = -1_000_000.0
            else:
                score = node.metric_value if node.maximize else -node.metric_value
                score += 0.01 / (1 + node.iteration)
            scored.append(
                {
                    "node_id": node.id,
                    "score": score,
                    "metric_value": node.metric_value,
                    "status": node.status,
                    "work_dir": node.work_dir,
                    "selected_for_expansion": False,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        for item in scored[: min(3, len(scored))]:
            item["selected_for_expansion"] = True
        return scored

    def _bfts_frontier_markdown(self, journal: ResearchJournal) -> str:
        frontier = self._bfts_frontier(journal)
        lines = ["# BFTS Frontier", ""]
        if not frontier:
            return "# BFTS Frontier\n\nNo frontier nodes were available.\n"
        for item in frontier:
            lines.append(f"## Node {item['node_id']}")
            lines.append(f"- score: {item['score']:.6g}")
            lines.append(f"- metric_value: {item['metric_value'] if item['metric_value'] is not None else 'n/a'}")
            lines.append(f"- status: {item['status']}")
            lines.append(f"- selected_for_expansion: {item['selected_for_expansion']}")
            lines.append(f"- work_dir: `{item['work_dir']}`")
            lines.append("")
        return "\n".join(lines)
