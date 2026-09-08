from __future__ import annotations

from src.agent_system.action import AgentAction
from src.agent_system.base_agent import BaseResearchAgent
from src.agent_system.state import ResearchState


ACTION_SCHEMA_PROMPT = (
    "Return one JSON object with fields: action_type, rationale, content, "
    "target_artifact, expected_effect, safety_checks, requires_human_review. "
    "Allowed action_type values: propose_hypothesis, revise_model, design_experiment, "
    "debug_error, propose_ablation, summarize_evidence, request_human_check, write_note."
)


class LLMPlannerAgent(BaseResearchAgent):
    name = "llm_planner"
    role = "plans research hypotheses and next actions"

    def decide(self, state: ResearchState) -> AgentAction:
        fallback = AgentAction(
            action_type="propose_hypothesis",
            rationale="Create a testable hypothesis before running autonomous experiments.",
            content=f"A computational OR method can improve the configured metric for: {state.research_goal}",
            expected_effect="Adds a concrete hypothesis to ResearchState.",
            safety_checks=["Human must verify novelty and formulation."],
        )
        return self._json_action("You are a cautious OR research planner. " + ACTION_SCHEMA_PROMPT, state.brief(), fallback)


class LLMModelCriticAgent(BaseResearchAgent):
    name = "llm_model_critic"
    role = "critiques assumptions and model risks"

    def decide(self, state: ResearchState) -> AgentAction:
        fallback = AgentAction(
            action_type="request_human_check",
            rationale="Model correctness cannot be delegated entirely to automation.",
            content="Verify objective direction, feasibility, variable domains, missing capacity/balance constraints, and baseline fairness.",
            target_artifact="human_gate_requests.md",
            expected_effect="Adds explicit human verification gate.",
            safety_checks=["No mathematical correctness is claimed automatically."],
        )
        return self._json_action("You are a skeptical OR model critic. " + ACTION_SCHEMA_PROMPT, state.brief(), fallback)


class LLMExperimentDesignerAgent(BaseResearchAgent):
    name = "llm_experiment_designer"
    role = "designs executable experiments and ablations"

    def decide(self, state: ResearchState) -> AgentAction:
        if state.metadata.get("latest_iteration_summary"):
            content = (
                "After observing the latest autonomous results: "
                "run baseline strengthening if proposed evidence is weak; "
                "run ablation if proposed evidence is positive; "
                "run adversarial stress test and replication before making claims."
            )
        else:
            content = (
                "Baseline strengthening branch; "
                "Ablation branch to weaken proposed component; "
                "Adversarial stress test branch with larger instances/noise; "
                "Replication branch with more seeds or repetitions."
            )
        fallback = AgentAction(
            action_type="design_experiment",
            rationale="A useful autonomous run needs baselines, stress tests, and ablations.",
            content=content,
            target_artifact="agent_action_plan.md",
            expected_effect="Guides autonomous search toward evidence-building rather than single-run optimization.",
            safety_checks=["Experiment must preserve results.csv contract.", "Runtime must stay within timeout."],
        )
        return self._json_action("You are an OR experiment designer. " + ACTION_SCHEMA_PROMPT, state.brief(), fallback)


class LLMEvidenceAgent(BaseResearchAgent):
    name = "llm_evidence"
    role = "summarizes evidence and claim boundaries"

    def decide(self, state: ResearchState) -> AgentAction:
        fallback = AgentAction(
            action_type="summarize_evidence",
            rationale="Research claims need explicit evidence boundaries.",
            content="Summarize whether current computational evidence supports, weakens, or leaves unresolved the hypothesis; mark all claims requiring human verification.",
            target_artifact="llm_agent_notes.md",
            expected_effect="Improves claim discipline before report generation.",
            safety_checks=["Do not claim novelty.", "Do not claim correctness without human review."],
        )
        return self._json_action("You are a conservative evidence reviewer. " + ACTION_SCHEMA_PROMPT, state.brief(), fallback)
