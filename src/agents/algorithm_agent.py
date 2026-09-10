"""An algorithm plan built from this study, or an admitted absence.

This agent used to ignore both of its arguments and return one fixed plan:
dynamic programming over integer weights, a value-density greedy, and one-for-one
swaps that preserve capacity. That is a knapsack algorithm, and it was proposed
for every research goal the system was ever given -- bilevel programming,
scheduling, inventory, anything. The plan read as if it had been chosen.

What can honestly be derived here is narrower than a plan: the domain profile
names algorithm families and metrics for the selected domain, and the experiment
template that will actually run declares what it implements. Those are real
inputs and they produce a real starting point. What cannot be derived is the
choice *between* those families for this particular problem -- that needs
something that has read the model. So when no model is available, this says so
rather than picking one and sounding certain.
"""

from __future__ import annotations

from src.core.artifact_provenance import ContentSource, declared_gap
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError
from src.schemas import AlgorithmPlan, ResearchIdea


ALGORITHM_SYSTEM_PROMPT = (
    "You are an Operations Research methodologist. You propose algorithms that fit the "
    "formulation you were given, you name baselines strong enough that beating them means "
    "something, and you say when a choice cannot be made from the information provided."
)


class AlgorithmAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        idea: ResearchIdea,
        model_markdown: str,
        domain_profile: dict | None = None,
        template_description: str = "",
    ) -> tuple[AlgorithmPlan, ContentSource, list[str]]:
        """Returns (plan, content_source, inputs_used)."""

        profile = domain_profile or {}
        failure = ""
        if not self.llm.use_mock:
            plan, failure = self._llm_plan(idea, model_markdown, profile)
            if plan is not None:
                return plan, "llm", ["selected_idea", "model_draft", "domain_profile"]

        families = [str(item) for item in profile.get("algorithm_families", []) or []]
        metrics = [str(item) for item in profile.get("metrics", []) or []]
        inputs_used = ["selected_idea"]
        if families or metrics:
            inputs_used.append("domain_profile")

        gap = declared_gap(
            "The choice of algorithm for this problem",
            (
                f"The model call failed ({failure}), so nothing read the formulation to choose "
                "between the candidate families below."
                if failure
                else "No language model was available to read the formulation and choose between "
                "the candidate families below."
            )
            + " Naming one anyway would be a guess presented as a plan -- "
            "the previous version of this stage proposed knapsack dynamic programming for every "
            "study, whatever the problem was.",
            [
                "a configured language model that has read the model draft, or",
                "a researcher who selects the family and states why it fits this formulation",
            ],
        )

        candidates = ", ".join(families) if families else "none are declared for this domain"
        plan = AlgorithmPlan(
            exact_solver_plan=(
                f"Candidate exact approaches for the `{profile.get('name', 'unknown')}` domain: "
                f"{candidates}. Not yet selected.\n\n{gap}"
            ),
            heuristic_plan=(
                f"The proposed algorithm type recorded on the selected idea is "
                f"`{idea.proposed_algorithm_type}`, and the proposed model type is "
                f"`{idea.proposed_model_type}`. Neither has been turned into a procedure, and "
                "neither was checked against the formulation."
            ),
            baseline_methods=families or ["NOT SELECTED - no algorithm family is declared for this domain"],
            pseudocode=(
                "NOT GENERATED. What actually runs is the experiment template selected by the "
                f"template registry{f' ({template_description})' if template_description else ''}, "
                "not a procedure derived from this plan. Read that template to see the algorithm "
                "the results actually came from."
            ),
            complexity_discussion=(
                "NOT ANALYSED. Complexity depends on the algorithm chosen, and none has been chosen."
            ),
            stopping_criteria=(
                "Determined by the experiment template and the run budget "
                "(`solver_timeout_seconds`, `max_research_iterations`), not by this plan."
            ),
            required_packages=["pandas", "numpy", "matplotlib"],
            evaluation_metrics=metrics or ["objective"],
        )
        return plan, "derived", inputs_used

    def _llm_plan(
        self, idea: ResearchIdea, model_markdown: str, profile: dict
    ) -> tuple[AlgorithmPlan | None, str]:
        try:
            payload = self.llm.chat_json(
                ALGORITHM_SYSTEM_PROMPT,
                (
                    "Propose an algorithm plan for this formulation. Return JSON with keys: "
                    "exact_solver_plan, heuristic_plan, baseline_methods (list), pseudocode, "
                    "complexity_discussion, stopping_criteria, required_packages (list), "
                    "evaluation_metrics (list). If the formulation is too underspecified to "
                    "choose an algorithm, say so in exact_solver_plan rather than guessing.\n\n"
                    f"Hypothesis: {idea.core_hypothesis}\n"
                    f"Proposed model type: {idea.proposed_model_type}\n"
                    f"Domain algorithm families: {', '.join(str(x) for x in profile.get('algorithm_families', []) or [])}\n\n"
                    f"Model draft:\n{model_markdown[:6000]}"
                ),
            )
            return AlgorithmPlan(**payload), ""
        except LLMCallError as error:
            if error.is_permanent:
                raise
            return None, error.summary()
        except Exception as exc:
            return None, f"the plan response could not be used: {exc}"
