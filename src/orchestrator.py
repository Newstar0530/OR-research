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
from src.core.method_registry import build_method_registry
from src.core.metric_registry import build_metric_registry
from src.core.problem_schema import ResearchArtifactManifest, ResearchProblem, infer_problem_family
from src.core.research_protocol import default_or_protocol
from src.core.model_ir import write_model_ir
from src.core.model_exporter import export_model_skeletons
from src.core.solver_adapter import detect_solver_tools, recommended_tools_for_profile
from src.domain_profiles import load_domain_profile, render_profile_markdown
from src.execution.experiment_logger import ExperimentLogger
from src.execution.sandbox_runner import SandboxRunner
from src.llm_client import LLMClient
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
from src.utils.research_memory import append_memory
from src.utils.run_tracker import RunTracker, write_run_status
from src.utils.semantic_scholar import write_semantic_scholar_report


class ResearchOrchestrator:
    stage_order = [
        "domain_classification",
        "idea_generation",
        "novelty_checking",
        "gap_synthesis",
        "mathematical_modeling",
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
        self.llm = LLMClient(config.llm_provider, config.model_name, config.use_mock_llm, tracker=self.llm_tracker)
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
            background_text=profile_markdown,
        )
        self._mark("idea_generation")
        write_json(run_dir / "idea_archive.json", idea_archive.model_dump())
        selected = idea_archive.ideas[idea_archive.selected_index]
        write_json(run_dir / "selected_idea.json", selected.model_dump())

        novelty = NoveltyAgent(self.llm).run(selected, local_literature=local_literature) if self.config.enable_novelty_check else None
        self._mark("novelty_checking")
        if novelty:
            (run_dir / "novelty_report.md").write_text(self._novelty_markdown(novelty), encoding="utf-8")
            if self.config.enable_semantic_scholar:
                write_semantic_scholar_report(novelty.search_queries, run_dir / "semantic_scholar_report.md")
        else:
            (run_dir / "novelty_report.md").write_text("# Novelty Report\n\nNovelty check disabled.\n", encoding="utf-8")

        gap_analysis = GapSynthesisAgent(self.llm).run(self.config.research_goal, domain_profile, literature_index_path)
        self._mark("gap_synthesis")
        (run_dir / "research_gap_analysis.md").write_text(gap_analysis, encoding="utf-8")

        template_text = read_text_if_exists(self._resolve_path(self.config.template_path))
        model = ModelingAgent(self.llm).run(selected, self.config.research_goal, template_text)
        self._mark("mathematical_modeling")
        (run_dir / "model_draft.md").write_text(model.markdown, encoding="utf-8")
        model_ir = write_model_ir(model.markdown, run_dir, self.config.project_name)
        export_model_skeletons(model_ir, run_dir)

        critique = CriticAgent(self.llm).run(model.markdown)
        self._mark("model_critique")
        (run_dir / "model_critique.md").write_text(self._critique_markdown(critique), encoding="utf-8")

        algorithm = AlgorithmAgent(self.llm).run(selected, model.markdown)
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
            )
            journal = search.run(
                run_dir,
                selected,
                algorithm,
                experiment.code,
                research_state=research_state,
                agent_runtime=runtime,
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
        evaluate_statistical_evidence(
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

        review = ReviewerAgent(self.llm).run(report_text)
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
            },
        )
        return run_dir

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
- Soundness: {r.soundness_score}/10
- Novelty: {r.novelty_score}/10
- Technical quality: {r.technical_quality_score}/10
- Reproducibility: {r.reproducibility_score}/10
- Presentation: {r.presentation_score}/10
- Overall: {r.overall_score}/10
- Confidence: {r.confidence_score}/10

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
