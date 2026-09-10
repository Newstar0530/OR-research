from __future__ import annotations

from src.llm_client import LLMClient
from src.llm_errors import LLMCallError
from src.schemas import IdeaArchive, ResearchIdea


class IdeaAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        research_goal: str,
        domain: str,
        max_ideas: int = 3,
        background_text: str = "",
        prior_findings: str = "",
    ) -> IdeaArchive:
        """Generate ideas, informed by what earlier runs on this goal measured.

        `prior_findings` is passed through to the model as evidence from those
        runs, explicitly not as settled fact -- an idea ruled out by one
        underpowered comparison is exactly the kind of idea worth retesting with
        a sharper design.
        """

        if not self.llm.use_mock:
            try:
                payload = self.llm.chat_json(
                    "You are an ambitious but skeptical Industrial Engineering / Operations Research PhD student.",
                    (
                        "Generate testable computational research ideas for this goal. "
                        "Reject vague ideas. Return JSON with key 'ideas', a list of objects matching these fields: "
                        "title, problem_context, core_hypothesis, expected_contribution, proposed_model_type, "
                        "proposed_algorithm_type, experimental_plan, interestingness_score, novelty_score, "
                        "feasibility_score, risk_score, assumptions, required_data, expected_outputs. "
                        f"Goal: {research_goal}\nDomain: {domain}\nBackground: {background_text[:4000]}"
                        + (f"\n\n{prior_findings[:2000]}" if prior_findings else "")
                    ),
                )
                ideas = [ResearchIdea(**item) for item in payload.get("ideas", [])[:max_ideas]]
                if ideas:
                    return IdeaArchive(ideas=ideas, selected_index=0)
            except LLMCallError as error:
                # A bad key or a wrong model name will fail identically at
                # every later stage. Degrading here would hide the cause six
                # times over.
                if error.is_permanent:
                    raise
            except Exception:
                pass

        profile = f"{domain}\n{background_text}".lower()
        if "scheduling" in profile:
            ideas = [self._family_idea(research_goal, domain, "scheduling", "time-indexed or disjunctive scheduling model", "dispatching baseline plus improvement heuristic")]
        elif "inventory" in profile:
            ideas = [self._family_idea(research_goal, domain, "inventory control", "stochastic inventory or lot-sizing model", "simulation and policy-search baseline")]
        elif "network" in profile:
            ideas = [self._family_idea(research_goal, domain, "network analysis", "graph, flow, or reliability model", "graph algorithm baseline plus disruption/sensitivity analysis")]
        elif "stochastic" in profile:
            ideas = [self._family_idea(research_goal, domain, "stochastic systems", "queueing, simulation, or stochastic programming model", "Monte Carlo scenario analysis")]
        elif "decision" in profile:
            ideas = [self._family_idea(research_goal, domain, "decision analysis", "multi-criteria decision model", "weighted scoring or robustness comparison")]
        elif "heuristic" in profile or "metaheuristic" in profile:
            ideas = [self._family_idea(research_goal, domain, "heuristic optimization", "combinatorial optimization model", "greedy, local-search, and metaheuristic baselines")]
        else:
            ideas = [self._family_idea(research_goal, domain, "optimization", "generic mathematical optimization model", "baseline versus proposed computational method")]
        return IdeaArchive(ideas=ideas[:max_ideas], selected_index=0)

    def _family_idea(self, goal: str, domain: str, family: str, model: str, algorithm: str) -> ResearchIdea:
        return ResearchIdea(
            title=f"Inspectable {family} research workflow with explicit baselines",
            problem_context=f"{domain}: {goal}",
            core_hypothesis=f"A transparent {algorithm} can reveal measurable trade-offs for the stated {family} research problem.",
            expected_contribution="A reproducible, human-verifiable OR experiment with explicit assumptions, baselines, metrics, sensitivity analysis, and cautious claims.",
            proposed_model_type=model,
            proposed_algorithm_type=algorithm,
            experimental_plan=(
                "Create or adapt a small executable experiment, compare at least two methods, "
                "record objective/runtime/feasibility metrics, and run parameter sensitivity checks."
            ),
            interestingness_score=7,
            novelty_score=5,
            feasibility_score=8,
            risk_score=4,
            assumptions=[
                "Synthetic or user-provided benchmark data is acceptable for the first validation.",
                "The generated model is a draft and requires human mathematical verification.",
                "Novelty must be checked against verified literature before making contribution claims.",
            ],
            required_data=["Research goal or abstract", "Optional literature folder", "Optional starter code or benchmark data"],
            expected_outputs=["results.csv", "figures/", "result_evaluation.md", "sensitivity_report.md", "final_report.md"],
        )

