from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal runtimes
    yaml = None

from src.agents.algorithm_agent import AlgorithmAgent
from src.agents.critic_agent import CriticAgent
from src.agents.debug_agent import DebugAgent
from src.agents.domain_classifier_agent import DomainClassifierAgent
from src.agents.experiment_agent import ExperimentAgent
from src.agents.gap_synthesis_agent import GapSynthesisAgent
from src.agents.idea_agent import IdeaAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.novelty_agent import NoveltyAgent
from src.agents.report_agent import ReportAgent
from src.agents.requirement_agent import RequirementAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.readiness_remediation_agent import ReadinessRemediationAgent
from src.agents.sensitivity_agent import SensitivityAgent
from src.agent_system.agents import LLMExperimentDesignerAgent, LLMEvidenceAgent, LLMModelCriticAgent, LLMPlannerAgent
from src.agent_system.policy import AgentPolicy
from src.agent_system.runtime import AgentRuntime
from src.agent_system.state import ResearchState
from src.config import AppConfig
from src.core.budget import ResearchBudget
from src.core.benchmark_loader import build_benchmark_manifest
from src.core.decision_engine import DecisionEngine
from src.core.domain_tools import get_domain_tool_spec
from src.core.experiment_contract import validate_experiment_contract
from src.core.experiment_search import ExperimentSearchController
from src.core.search_policy import SearchPolicyConfig
from src.core.method_registry import build_method_registry
from src.core.metric_registry import build_metric_registry
from src.core.problem_schema import ResearchArtifactManifest, ResearchProblem, infer_problem_family
from src.core.ir_compiler import compile_to_mip
from src.core.mixed_integer_program import solve_mip, verify_mip_solution
from src.core.requirement_coverage import coverage_markdown
from src.core.research_protocol import default_or_protocol
from src.core.model_ir import build_model_ir, write_model_ir
from src.core.model_exporter import export_model_skeletons
from src.core.solver_adapter import detect_solver_tools, recommended_tools_for_profile
from src.domain_profiles import load_domain_profile, render_profile_markdown
from src.execution.experiment_logger import ExperimentLogger
from src.execution.sandbox_runner import SandboxRunner
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError
from src.utils.file_utils import ensure_dir, read_text_if_exists, write_json
from src.utils.autonomy_readiness import evaluate_autonomy_readiness
from src.utils.claim_checker import check_report_claims
from src.utils.citation_manager import build_citation_artifacts
from src.utils.bfts_tree import build_research_tree
from src.utils.literature import build_literature_index
from src.utils.literature_grounding import build_literature_grounding
from src.utils.latex_writer import write_latex_paper
from src.utils.latex_compile import compile_latex, review_pdf_stub
from src.utils.llm_tracker import LLMInteractionTracker, write_llm_tracker
from src.utils.plotting import save_objective_gap_plot
from src.utils.result_evaluator import evaluate_results
from src.utils.statistical_evidence import evaluate_statistical_evidence
from src.core.artifact_provenance import ProvenanceLedger, write_provenance
from src.utils.bibliography import literature_digest
from src.utils.research_memory import (
    RunRecord,
    append_memory,
    build_prior_findings,
    record_run,
    summarize_journal,
    write_prior_findings,
)
from src.utils.run_tracker import RunTracker, write_run_status
from src.utils.semantic_scholar import write_semantic_scholar_report


class ResearchOrchestrator:
    stage_order = [
        "domain_classification",
        "idea_generation",
        "novelty_checking",
        "gap_synthesis",
        "requirement_extraction",
        "mathematical_modeling",
        "requirement_coverage",
        "model_compilation",
        "model_critique",
        "algorithm_proposal",
        "experiment_implementation",
        "experiment_execution_and_debugging",
        "sensitivity_analysis",
        "report_generation",
        "automated_review",
    ]

    def __init__(self, config: AppConfig, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = project_root or Path.cwd()
        self.llm_tracker = LLMInteractionTracker()
        self.llm = LLMClient(
            config.llm_provider,
            config.model_name,
            config.use_mock_llm,
            tracker=self.llm_tracker,
            max_attempts=config.llm_max_attempts,
            backoff_seconds=config.llm_backoff_seconds,
            timeout_seconds=config.llm_timeout_seconds,
            temperature=config.llm_temperature,
            max_calls=config.max_llm_calls,
            max_cost_usd=config.max_llm_cost_usd,
            price_per_1k_prompt_tokens=config.llm_price_per_1k_prompt_tokens,
            price_per_1k_response_tokens=config.llm_price_per_1k_response_tokens,
        )
        self.completed_stages: list[str] = []

    def run(self) -> Path:
        run_dir = self._create_run_dir()
        tracker = RunTracker.started()
        write_run_status(run_dir, tracker)
        figures_dir = ensure_dir(run_dir / "figures")
        config_text = yaml.safe_dump(self.config.model_dump(), sort_keys=False) if yaml else self._simple_yaml_dump(self.config.model_dump())
        (run_dir / "config_used.yaml").write_text(config_text, encoding="utf-8")
        protocol = default_or_protocol()
        (run_dir / "research_protocol.md").write_text(protocol.to_markdown(), encoding="utf-8")
        write_json(run_dir / "artifact_manifest.json", ResearchArtifactManifest().model_dump())
        write_json(run_dir / "method_registry.json", [m.model_dump() for m in build_method_registry().values()])
        write_json(run_dir / "metric_registry.json", [m.model_dump() for m in build_metric_registry().values()])
        build_benchmark_manifest(self.config.benchmark_dir, run_dir)
        solver_report = detect_solver_tools()
        write_json(run_dir / "solver_tool_report.json", solver_report.model_dump())
        (run_dir / "solver_tool_report.md").write_text(solver_report.to_markdown(), encoding="utf-8")
        logger = ExperimentLogger(run_dir)
        local_literature = []
        literature_index_path = None
        if self.config.literature_dir:
            literature_index, _ = build_literature_index(self.config.literature_dir, run_dir)
            literature_index_path = literature_index
            try:
                lit_df = pd.read_csv(literature_index)
                if "keyword_hits" in lit_df.columns:
                    lit_df["relevance_score"] = lit_df["keyword_hits"].fillna("").apply(
                        lambda s: len([term for term in str(s).split(",") if term.strip()])
                    )
                    local_literature = lit_df.sort_values(["relevance_score", "filename"], ascending=[False, True]).head(20).to_dict("records")
            except Exception:
                local_literature = []
        # The uploaded papers used to stop here: `background_text` for ideation
        # was the static domain-profile YAML, so a user who supplied literature
        # got ideas that had never seen it.
        literature_brief = literature_digest(local_literature)

        # Read the project's memory before any planning agent runs, so what the
        # earlier runs measured is available to the agents that choose what to
        # try -- not merely reported after the fact.
        self._preflight_llm(run_dir)
        prior_findings = build_prior_findings(
            self.project_root,
            self.config.research_goal,
            exclude_run_dirs=[str(run_dir)],
        )
        write_prior_findings(run_dir, prior_findings)
        prior_brief = prior_findings.to_prompt()
        # Every front-half stage declares where its content came from, so a
        # placeholder can never be mistaken for a finding.
        provenance = ProvenanceLedger()

        domain_selection = DomainClassifierAgent(self.llm).run(self.config.research_goal, self.config.research_domain_mode)
        domain_profile = load_domain_profile(domain_selection["profile"])
        # When the config pins an experiment template, the template -- not the
        # topic classifier -- decides which tools, metrics and mutations apply.
        experiment_profile = self.config.experiment_template or domain_selection["profile"]
        domain_tool_spec = get_domain_tool_spec(experiment_profile)
        profile_markdown = render_profile_markdown(domain_profile)
        problem = ResearchProblem(
            title=self.config.project_name,
            goal=self.config.research_goal,
            domain=self.config.domain,
            problem_family=infer_problem_family(domain_selection["profile"]),
            objective_candidates=domain_profile.get("metrics", [])[:5],
            controllable_factors=domain_profile.get("sensitivity_parameters", [])[:5],
            baseline_candidates=domain_profile.get("algorithm_families", [])[:5],
            human_verification_needed=domain_profile.get("human_checks", [])[:5],
        )
        self._mark("domain_classification")
        provenance.record(
            "domain_classification",
            "derived",
            domain_selection.get("reason", ""),
            inputs_used=["research_goal"],
        )
        write_json(run_dir / "domain_selection.json", domain_selection)
        write_json(
            run_dir / "recommended_solver_tools.json",
            {"tools": recommended_tools_for_profile(domain_selection["profile"], solver_report)},
        )
        write_json(run_dir / "domain_tool_spec.json", domain_tool_spec.model_dump())
        (run_dir / "domain_tool_spec.md").write_text(self._domain_tool_markdown(domain_tool_spec), encoding="utf-8")
        write_json(run_dir / "problem_schema.json", problem.model_dump())
        (run_dir / "domain_profile.md").write_text(profile_markdown, encoding="utf-8")
        research_state = None
        runtime = None
        if self.config.enable_llm_agent_runtime:
            research_state = ResearchState(
                project_name=self.config.project_name,
                research_goal=self.config.research_goal,
                domain_profile=domain_selection["profile"],
                run_dir=str(run_dir),
                artifacts={
                    "domain_selection": "domain_selection.json",
                    "problem_schema": "problem_schema.json",
                    "domain_profile": "domain_profile.md",
                },
            )
            runtime = AgentRuntime(
                [
                    LLMPlannerAgent(self.llm),
                    LLMModelCriticAgent(self.llm),
                    LLMExperimentDesignerAgent(self.llm),
                    LLMEvidenceAgent(self.llm),
                ],
                AgentPolicy(run_dir),
            )
            research_state = runtime.run_round(research_state)
            runtime.save(run_dir, research_state)

        idea_archive = IdeaAgent(self.llm).run(
            self.config.research_goal,
            self.config.domain,
            self.config.max_ideas,
            background_text=(
                profile_markdown + ("\n\n" + literature_brief if literature_brief else "")
            ),
            prior_findings=prior_brief,
        )
        provenance.record(
            "idea_generation",
            "derived" if self.llm.use_mock else "llm",
            "Mock mode selects a template idea by domain keyword; no model reasoned about the goal."
            if self.llm.use_mock
            else "A language model generated the idea archive.",
            inputs_used=["research_goal", "domain_profile"]
            + (["literature_index"] if literature_brief else [])
            + (["prior_findings"] if prior_brief else []),
        )
        self._mark("idea_generation")
        write_json(run_dir / "idea_archive.json", idea_archive.model_dump())
        selected = idea_archive.ideas[idea_archive.selected_index]
        write_json(run_dir / "selected_idea.json", selected.model_dump())

        novelty_source = None
        if self.config.enable_novelty_check:
            novelty, novelty_source = NoveltyAgent(self.llm).run(selected, local_literature=local_literature)
        else:
            novelty = None
        provenance.record(
            "novelty_checking",
            novelty_source or "not_generated",
            "Search queries were built from the idea; no database was queried."
            if novelty_source
            else "The novelty check was disabled for this run.",
            inputs_used=["selected_idea"] if novelty_source else [],
            missing=[] if novelty_source else ["enable_novelty_check: true"],
        )
        self._mark("novelty_checking")
        if novelty:
            (run_dir / "novelty_report.md").write_text(self._novelty_markdown(novelty), encoding="utf-8")
            if self.config.enable_semantic_scholar:
                write_semantic_scholar_report(novelty.search_queries, run_dir / "semantic_scholar_report.md")
        else:
            (run_dir / "novelty_report.md").write_text("# Novelty Report\n\nNovelty check disabled.\n", encoding="utf-8")

        gap_analysis, gap_source, gap_inputs = GapSynthesisAgent(self.llm).run(
            self.config.research_goal, domain_profile, literature_index_path
        )
        provenance.record(
            "gap_synthesis",
            gap_source,
            "A model proposed the gaps from the indexed evidence."
            if gap_source == "llm"
            else "No model was available, so no research gap was proposed.",
            inputs_used=gap_inputs if gap_source != "not_generated" else [],
            missing=[]
            if gap_source != "not_generated"
            else ["a configured LLM, or a researcher who has read the literature"],
        )
        self._mark("gap_synthesis")
        (run_dir / "research_gap_analysis.md").write_text(gap_analysis, encoding="utf-8")

        # What the formulation will be held to, written down before it exists so
        # the draft is measured against something not derived from the draft.
        requirements, requirement_source = RequirementAgent(self.llm).run(
            self.config.research_goal, selected, domain_profile
        )
        provenance.record(
            "requirement_extraction",
            requirement_source,
            f"{len(requirements.requirements)} requirement(s) were extracted from the research goal"
            " and the selected idea; the formulation is checked against them."
            if requirements.requirements
            else "No requirement could be extracted, so the formulation has nothing to be checked"
            " against and its completeness is unknown rather than confirmed.",
            inputs_used=["research_goal", "selected_idea"] if requirements.requirements else [],
            missing=[] if requirements.requirements else ["a research goal that states its conditions"],
        )
        self._mark("requirement_extraction")
        write_json(run_dir / "requirement_set.json", requirements.model_dump())
        problem.constraint_candidates = [item.text for item in requirements.constraint_requirements]
        write_json(run_dir / "problem_schema.json", problem.model_dump())

        template_text = read_text_if_exists(self._resolve_path(self.config.template_path))
        model, model_source, model_inputs, coverage = ModelingAgent(self.llm).run(
            selected,
            self.config.research_goal,
            template_text,
            domain_profile=domain_profile,
            literature_brief=literature_brief,
            requirements=requirements,
        )
        provenance.record(
            "mathematical_modeling",
            model_source,
            (
                "A language model declared the formulation as typed fields: every variable has a"
                " domain, every constraint an expression, and `model_draft.md` is rendered from"
                " those fields rather than written alongside them."
                if model.model is not None
                else "A language model wrote the formulation as prose and it passed inspection:"
                " no scaffold placeholders, all sections present, real mathematics, and a chosen"
                " objective direction. Units, bounds and requirement links exist only where the"
                " prose happened to state them."
            )
            if model_source == "llm"
            else "`model_draft.md` is the generic scaffold with this run's goal interpolated into"
            " it. No model formulated this problem; see `model_critique.md` for which elements"
            " are still placeholders.",
            inputs_used=model_inputs if model_source == "llm" else [],
            missing=[]
            if model_source == "llm"
            else ["a configured LLM, or a researcher writing the formulation"],
        )
        self._mark("mathematical_modeling")
        (run_dir / "model_draft.md").write_text(model.markdown, encoding="utf-8")
        # A formulation that came back as fields is used as it was declared.
        # One that only ever existed as prose is scraped back out of it, which
        # recovers far less -- and says so in its own provenance.
        model_ir = model.model or build_model_ir(model.markdown, self.config.project_name)
        write_model_ir(model_ir, run_dir)
        export_model_skeletons(model_ir, run_dir)

        provenance.record(
            "requirement_coverage",
            "derived" if coverage.checked else "not_generated",
            f"{len(coverage.encoded)} of {len(coverage.matches)} requirements appear to be imposed"
            f" by a constraint; {len(coverage.missing)} are not."
            if coverage.checked
            else "There was no requirement list to check the formulation against.",
            inputs_used=["model_draft", "requirement_set"] if coverage.checked else [],
            missing=[] if coverage.checked else ["an extracted requirement list"],
        )
        self._mark("requirement_coverage")
        write_json(run_dir / "requirement_coverage.json", coverage.model_dump())
        (run_dir / "requirement_coverage.md").write_text(coverage_markdown(coverage), encoding="utf-8")

        # The one place the research pipeline reaches the verified solver path.
        # A formulation that compiles is solved, re-substituted into its own
        # constraints and cross-checked; one that does not says what is missing.
        compilation = compile_to_mip(model_ir)
        provenance.record(
            "model_compilation",
            "derived" if compilation.compiled else "not_generated",
            f"The formulation compiled into an executable model with {compilation.mip.n_vars}"
            f" variables and {compilation.mip.n_constraints} constraints, and was solved and"
            " verified."
            if compilation.compiled and compilation.mip is not None
            else "The formulation did not compile; `model_compile_report.md` lists what each"
            " refusal needs.",
            inputs_used=["model_ir"],
            missing=[] if compilation.compiled else [item.reason for item in compilation.refusals][:3],
        )
        self._mark("model_compilation")
        write_json(
            run_dir / "model_compile_report.json", compilation.model_dump(exclude={"mip"})
        )
        (run_dir / "model_compile_report.md").write_text(compilation.markdown(), encoding="utf-8")
        if compilation.mip is not None:
            write_json(run_dir / "model_mip.json", compilation.mip.model_dump())
            self._solve_compiled_model(compilation.mip, run_dir)

        critique, critique_source = CriticAgent(self.llm).run(model.markdown, coverage)
        provenance.record(
            "model_critique",
            critique_source,
            "The draft was inspected for unfilled scaffold placeholders and missing sections, and"
            " compared against the requirement list.",
            inputs_used=["model_draft", "requirement_coverage"],
        )
        self._mark("model_critique")
        (run_dir / "model_critique.md").write_text(self._critique_markdown(critique), encoding="utf-8")

        algorithm, algorithm_source, algorithm_inputs = AlgorithmAgent(self.llm).run(
            selected,
            model.markdown,
            domain_profile=domain_profile,
            template_description=self.config.experiment_template or domain_selection["profile"],
        )
        provenance.record(
            "algorithm_proposal",
            algorithm_source,
            "A model chose the algorithm from the formulation."
            if algorithm_source == "llm"
            else "Candidate families come from the domain profile; none was selected for this"
            " problem. What actually runs is the experiment template.",
            inputs_used=algorithm_inputs,
            missing=[]
            if algorithm_source == "llm"
            else ["a configured LLM, or a researcher choosing the algorithm family"],
        )
        self._mark("algorithm_proposal")
        (run_dir / "algorithm_plan.md").write_text(self._algorithm_markdown(algorithm), encoding="utf-8")

        experiment = ExperimentAgent(self.llm).run(
            selected,
            algorithm,
            template_text,
            domain_profile=domain_selection["profile"],
            template_key=self.config.experiment_template,
        )
        self._mark("experiment_implementation")
        experiment_path = run_dir / "generated_experiment.py"
        experiment_path.write_text(experiment.code, encoding="utf-8")

        runner = SandboxRunner(self.config.solver_timeout_seconds)
        journal = None
        if self.config.enable_autonomous_loop:
            search = ExperimentSearchController(
                self.llm,
                runner,
                ResearchBudget(
                    max_iterations=self.config.max_research_iterations,
                    max_branches=self.config.max_candidate_branches,
                    timeout_seconds=self.config.solver_timeout_seconds,
                    patience=self.config.autonomous_patience,
                    min_improvement=self.config.min_metric_improvement,
                ),
                DecisionEngine(
                    self.config.primary_metric,
                    self.config.objective_direction,
                    metric_method=self.config.primary_metric_method,
                    baseline_method=self.config.baseline_method,
                    proposed_method=self.config.proposed_method,
                ),
                domain_profile=experiment_profile,
                code_editing_backend=self.config.code_editing_backend,
                aider_command=self.config.aider_command,
                aider_model=self.config.aider_model,
                aider_timeout_seconds=self.config.aider_timeout_seconds,
                max_repair_attempts=self.config.max_debug_attempts,
                llm_repair_attempts=self.config.llm_repair_attempts,
                search_policy=SearchPolicyConfig(
                    num_drafts=self.config.search_num_drafts,
                    max_debug_depth=self.config.search_max_debug_depth,
                    debug_probability=self.config.search_debug_probability,
                    exploration_weight=self.config.search_exploration_weight,
                    seed=self.config.search_seed,
                ),
            )
            journal = search.run(
                run_dir,
                selected,
                algorithm,
                experiment.code,
                research_state=research_state,
                agent_runtime=runtime,
                prior_findings=prior_brief,
            )
            best = journal.best_node()
            status = best.status if best else "failed"
            metrics = {self.config.primary_metric: best.metric_value} if best and best.metric_value is not None else {}
            logger.append("autonomous_experiment_search", status, experiment_path, metrics, notes=f"nodes={len(journal.nodes)}")
            if runtime is not None and research_state is not None:
                runtime.save(run_dir, research_state)
        else:
            execution, debug = DebugAgent(self.llm).run(
                experiment_path, run_dir, runner, self.config.max_debug_attempts
            )
            logger.append("experiment_execution", execution.status, experiment_path, execution.metrics, notes=debug.remaining_errors[:500])
        self._mark("experiment_execution_and_debugging")
        build_research_tree(run_dir / "autonomous_journal.json", run_dir, maximize=self.config.objective_direction == "maximize")
        contract_report = validate_experiment_contract(run_dir)
        (run_dir / "experiment_contract.md").write_text(contract_report.to_markdown(), encoding="utf-8")

        plot = save_objective_gap_plot(run_dir / "results.csv", figures_dir)
        if plot:
            logger.append("plotting", "success", experiment_path, plots=[str(plot.relative_to(run_dir))])

        sensitivity = SensitivityAgent(self.llm).run(run_dir / "results.csv")
        self._mark("sensitivity_analysis")
        (run_dir / "sensitivity_report.md").write_text(sensitivity.markdown, encoding="utf-8")
        result_evaluation = evaluate_results(run_dir / "results.csv")
        (run_dir / "result_evaluation.md").write_text(result_evaluation, encoding="utf-8")
        statistical_evidence = evaluate_statistical_evidence(
            run_dir / "results.csv",
            run_dir,
            metric=self.config.primary_metric,
            objective_direction=self.config.objective_direction,
            baseline_method=self.config.baseline_method,
            proposed_method=self.config.proposed_method,
        )

        if self.config.enable_report_generation:
            report = ReportAgent(self.llm).run(selected, model.markdown, critique, algorithm, sensitivity.markdown)
            report_text = report.markdown
        else:
            report_text = "# Final Report\n\nReport generation disabled.\n"
        self._mark("report_generation")
        (run_dir / "final_report.md").write_text(report_text, encoding="utf-8")
        build_citation_artifacts(literature_index_path, run_dir)
        (run_dir / "claim_check.md").write_text(check_report_claims(report_text, literature_index_path), encoding="utf-8")
        if literature_index_path:
            build_literature_grounding(run_dir / "final_report.md", literature_index_path, run_dir)
        else:
            write_json(
                run_dir / "literature_grounding.json",
                {"claims_checked": 0, "grounded_claims": 0, "weak_claims": 0, "records": []},
            )
            (run_dir / "literature_grounding.md").write_text(
                "# Literature Grounding\n\nNo literature index was available for grounding.\n",
                encoding="utf-8",
            )

        review, review_source = ReviewerAgent(self.llm).run(report_text)
        provenance.record(
            "automated_review",
            review_source,
            "A model reviewed the report and produced the scores."
            if review_source == "llm"
            else "Nothing read the report; the scores are absent rather than defaulted.",
            inputs_used=["final_report"] if review_source == "llm" else [],
            missing=[] if review_source == "llm" else ["a configured LLM, or a human reviewer"],
        )
        self._mark("automated_review")
        (run_dir / "automated_review.md").write_text(self._review_markdown(review), encoding="utf-8")
        write_latex_paper(run_dir)
        compile_latex(run_dir)
        review_pdf_stub(run_dir)
        readiness = evaluate_autonomy_readiness(run_dir)
        write_json(run_dir / "autonomy_readiness.json", readiness.model_dump())
        (run_dir / "autonomy_readiness.md").write_text(readiness.to_markdown(), encoding="utf-8")
        remediation = ReadinessRemediationAgent(self.llm).run(readiness, self.config)
        write_json(run_dir / "next_research_cycle_plan.json", remediation.model_dump())
        (run_dir / "next_research_cycle_plan.md").write_text(remediation.to_markdown(), encoding="utf-8")
        (run_dir / "next_config_patch.yaml").write_text(remediation.config_patch_yaml(), encoding="utf-8")
        # The ledger's own headline says whether a model reasoned about
        # anything; the tracker knows whether the calls succeeded.
        provenance.record(
            "llm_calls",
            "derived",
            self.llm_tracker.headline(),
            inputs_used=["llm_interactions"],
        )
        write_provenance(run_dir, provenance)
        write_json(run_dir / "stage_order.json", self.completed_stages)
        tracker.finish("completed", run_dir, self.completed_stages)
        write_run_status(run_dir, tracker)
        write_llm_tracker(self.llm_tracker, run_dir)
        append_memory(
            self.project_root,
            "experiments",
            {
                "project_name": self.config.project_name,
                "research_goal": self.config.research_goal,
                "run_dir": str(run_dir),
                "stages": self.completed_stages,
                "review_overall_score": review.overall_score,
                "review_performed": review.review_performed,
            },
        )
        self._record_run_memory(
            run_dir, journal, statistical_evidence, review, domain_selection["profile"]
        )
        return run_dir

    def _preflight_llm(self, run_dir: Path) -> None:
        """One probe call before the pipeline, so a bad key fails in seconds.

        Without this, a mistyped key degrades every LLM-backed stage in turn and
        the run finishes with a report saying no stage used a language model --
        true, and silent about the one thing worth knowing. A permanent error
        stops the run here and names what to change; a transient one is recorded
        and the run continues, since it may well clear.
        """

        if self.config.use_mock_llm or not self.config.llm_preflight:
            return
        error = self.llm.preflight()
        if error is None:
            return
        (run_dir / "llm_preflight.md").write_text(
            "# LLM Preflight\n\n"
            f"- provider: {self.config.llm_provider}\n"
            f"- model: {self.config.model_name}\n"
            f"- result: {error.summary()}\n"
            f"- permanent: {error.is_permanent}\n",
            encoding="utf-8",
        )
        if error.is_permanent:
            raise LLMCallError(
                error.kind,
                (
                    f"LLM preflight failed and the cause is permanent, so the run was stopped "
                    f"before doing any work: {error.summary()} "
                    "Set `llm_preflight: false` to run with every model-backed stage degraded on "
                    "purpose."
                ),
                status=error.status,
                detail=error.detail,
            )

    def _record_run_memory(
        self,
        run_dir: Path,
        journal,
        statistical_evidence,
        review,
        domain: str = "",
    ) -> None:
        """Write this run into the project's memory for the next run to read.

        Everything stored here is a measurement this run actually produced. A
        run whose search never started records zero nodes rather than nothing,
        because "we tried and got nowhere" is itself worth carrying forward.

        Memory is written last and failure to write it is not allowed to fail
        the run: the report is already on disk, and losing a memory line costs
        the next run some context, not this run its results.
        """

        try:
            summary = (
                summarize_journal(
                    journal,
                    primary_metric=self.config.primary_metric,
                    objective_direction=self.config.objective_direction,
                )
                if journal is not None
                else {
                    "primary_metric": self.config.primary_metric,
                    "objective_direction": self.config.objective_direction,
                }
            )
            notes = []
            if journal is None:
                notes.append("The autonomous search loop was disabled for this run.")
            record = RunRecord(
                project_name=self.config.project_name,
                research_goal=self.config.research_goal,
                run_dir=str(run_dir),
                domain=domain or self.config.domain,
                evidence_strength=getattr(statistical_evidence, "evidence_strength", "") or "",
                supports_improvement_claim=bool(
                    getattr(statistical_evidence, "supports_improvement_claim", False)
                ),
                refuted_hypotheses=self._unsupported_hypotheses(journal, statistical_evidence),
                # None when no review happened. The previous version stored a
                # constant 6 here, so memory carried a number nothing had measured.
                review_overall_score=getattr(review, "overall_score", None)
                if getattr(review, "review_performed", False)
                else None,
                notes=notes,
                **summary,
            )
            record_run(self.project_root, record)
        except Exception as exc:  # pragma: no cover - memory must never break a run
            (run_dir / "research_memory_error.txt").write_text(
                f"Could not write the cross-run memory record: {exc}\n", encoding="utf-8"
            )

    def _unsupported_hypotheses(self, journal, statistical_evidence) -> list[str]:
        """Hypotheses this run tested and did not support.

        `not supported` is the whole claim. The comparison may have been
        underpowered, the effect may be real and small, the instance family may
        have been wrong -- so the wording carries the run's own verdict rather
        than upgrading it to a refutation.
        """

        unsupported: list[str] = []
        strength = getattr(statistical_evidence, "evidence_strength", "") or "unrecorded"
        if statistical_evidence is not None and not getattr(
            statistical_evidence, "supports_improvement_claim", False
        ):
            unsupported.append(
                f"The proposed method did not beat the baseline on {self.config.primary_metric}"
                f" at the run's significance level (verdict `{strength}`)."
            )
        if journal is None:
            return unsupported
        for node in getattr(journal, "nodes", []) or []:
            comparison = (getattr(node, "metadata", {}) or {}).get("method_comparison")
            if isinstance(comparison, dict) and comparison.get("supports_hypothesis") is False:
                reason = comparison.get("reason") or "no reason recorded"
                unsupported.append(f"Node `{node.id}`: {reason}")
        return unsupported[:10]

    def _create_run_dir(self) -> Path:
        safe_project = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.config.project_name).strip("_") or "project"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ensure_dir(self.project_root / self.config.output_dir / f"{timestamp}_{safe_project}")

    def _resolve_path(self, path: str | None) -> Path | None:
        if not path:
            return None
        p = Path(path)
        return p if p.is_absolute() else self.project_root / p

    def _mark(self, stage: str) -> None:
        self.completed_stages.append(stage)

    @staticmethod
    def _simple_yaml_dump(data: dict) -> str:
        return "\n".join(f"{key}: {value!r}" for key, value in data.items()) + "\n"

    @staticmethod
    def _novelty_markdown(report) -> str:
        return f"""# Novelty Report

## Summary
{report.novelty_summary}

## Search Queries
{chr(10).join(f"- {q}" for q in report.search_queries)}

## Potentially Related Works
{chr(10).join(f"- {w}" for w in report.potentially_related_works) if report.potentially_related_works else "- No verified sources found."}

## Similarity Risk
{report.similarity_risk}

## Recommendation
{report.recommendation}

## Revision Suggestions
{chr(10).join(f"- {s}" for s in report.revision_suggestions)}
"""

    def _solve_compiled_model(self, mip, run_dir: Path) -> None:
        """Solve the compiled formulation and record the independent check.

        The recorded objective is the one recomputed from the returned vector,
        not the one the solver printed, and a solver that disagrees with the
        cross-check is reported rather than reconciled.
        """

        solution = solve_mip(mip, time_limit=float(self.config.solver_timeout_seconds))
        verification = (
            verify_mip_solution(mip, solution.values) if solution.has_solution else None
        )
        write_json(
            run_dir / "model_solution.json",
            {
                "status": solution.status,
                "backend": solution.backend,
                "runtime_seconds": solution.runtime_seconds,
                "reported_objective": solution.objective,
                "recomputed_objective": verification.recomputed_objective if verification else None,
                "feasible": verification.feasible if verification else None,
                "max_violation": verification.max_violation if verification else None,
                "values": dict(zip([v.name for v in mip.variables], solution.values)),
                "notes": solution.notes,
            },
        )
        lines = [
            "# Solved From The Formulation",
            "",
            "This is the model declared in `model_ir.json`, compiled and solved -- not a",
            "template with its own numbers.",
            "",
            f"- status: {solution.status}",
            f"- backend: {solution.backend}",
        ]
        if verification is not None:
            lines.extend(
                [
                    f"- objective as recomputed from the returned solution: "
                    f"{verification.recomputed_objective}",
                    f"- feasible when substituted back into every constraint: {verification.feasible}",
                    f"- largest constraint violation: {verification.max_violation:.6g}",
                    "",
                    "## Solution",
                    "",
                ]
            )
            lines.extend(
                f"- `{variable.name}` = {value:g}"
                for variable, value in zip(mip.variables, solution.values)
            )
        if solution.notes:
            lines.extend(["", "## Solver Notes", "", f"- {solution.notes}"])
        (run_dir / "model_solution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _critique_markdown(c) -> str:
        sections = {
            "Critical Issues": c.critical_issues,
            "Minor Issues": c.minor_issues,
            "Missing Constraints": c.missing_constraints,
            "Questionable Assumptions": c.questionable_assumptions,
            "Suggested Revisions": c.suggested_revisions,
            "Human Verification Checklist": c.human_verification_checklist,
        }
        return "# Model Critique\n\n" + "\n\n".join(f"## {k}\n" + "\n".join(f"- {x}" for x in v) for k, v in sections.items())

    @staticmethod
    def _algorithm_markdown(a) -> str:
        return f"""# Algorithm Plan

## Exact Solver Plan
{a.exact_solver_plan}

## Heuristic Plan
{a.heuristic_plan}

## Baselines
{chr(10).join(f"- {b}" for b in a.baseline_methods)}

## Pseudocode
{a.pseudocode}

## Complexity
{a.complexity_discussion}

## Stopping Criteria
{a.stopping_criteria}

## Required Packages
{chr(10).join(f"- {p}" for p in a.required_packages)}

## Evaluation Metrics
{chr(10).join(f"- {m}" for m in a.evaluation_metrics)}
"""

    @staticmethod
    def _review_markdown(r) -> str:
        return f"""# Automated Review

## Summary
{r.summary}

## Strengths
{chr(10).join(f"- {x}" for x in r.strengths)}

## Weaknesses
{chr(10).join(f"- {x}" for x in r.weaknesses)}

## Questions for Authors
{chr(10).join(f"- {x}" for x in r.questions_for_authors)}

## Scores
{_score_lines(r)}

## Recommendation
{r.recommendation}

## Required Revision Checklist
{chr(10).join(f"- {x}" for x in r.required_revision_checklist)}
"""

    @staticmethod
    def _domain_tool_markdown(spec) -> str:
        return f"""# Domain Tool Spec

- profile: {spec.profile}
- template_key: {spec.template_key}
- primary_metric: {spec.primary_metric}
- objective_direction: {spec.objective_direction}

## Metrics
{chr(10).join(f"- {m}" for m in spec.metrics)}

## Mutation Hints
{chr(10).join(f"- {m}" for m in spec.mutation_hints)}

## Human Checks
{chr(10).join(f"- {m}" for m in spec.human_checks)}
"""


def _score_lines(review) -> str:
    """Render the review scores, or say plainly that there are none.

    Printing `None/10` would read as a broken template; printing a default
    would read as a verdict. Neither is what happened."""

    fields = (
        ("Soundness", review.soundness_score),
        ("Novelty", review.novelty_score),
        ("Technical quality", review.technical_quality_score),
        ("Reproducibility", review.reproducibility_score),
        ("Presentation", review.presentation_score),
        ("Overall", review.overall_score),
        ("Confidence", review.confidence_score),
    )
    if not getattr(review, "review_performed", False) or all(value is None for _, value in fields):
        return (
            "No scores: no review was performed. An absent score is not a low score, "
            "and it is not a high one."
        )
    return "\n".join(
        f"- {label}: {value}/10" if value is not None else f"- {label}: not scored"
        for label, value in fields
    )
