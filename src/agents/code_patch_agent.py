from __future__ import annotations

from src.agent_system.patch import PatchOperation, PatchProposal
from src.core.mutation import MutationSpec
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError


class CodePatchAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, code: str, plan: str, mutation: MutationSpec) -> PatchProposal | None:
        if not self.llm.use_mock:
            try:
                payload = self.llm.chat_json(
                    "You are a cautious OR coding agent. Propose only safe exact string replacements for experiment.py.",
                    (
                        "Return JSON matching PatchProposal: target_file, rationale, operations, expected_effect, safety_checks. "
                        "Each operation must have find, replace, occurrence, rationale. Preserve results.csv and SUMMARY_JSON. "
                        f"Plan: {plan}\nMutation: {mutation.model_dump()}\nCode excerpt:\n{code[:4000]}"
                    ),
                )
                return PatchProposal(**payload)
            except LLMCallError as error:
                if error.is_permanent:
                    raise
                return None
            except Exception:
                return None
        if mutation.kind == "baseline_strengthening" and "BASELINE_EFFECT = 0.0" in code:
            return PatchProposal(
                rationale="Strengthen baseline so proposed-method evidence is tested against a less naive comparator.",
                operations=[
                    PatchOperation(
                        find="BASELINE_EFFECT = 0.0",
                        replace="BASELINE_EFFECT = 0.3",
                        rationale="Give baseline a small improvement allowance.",
                    )
                ],
                expected_effect="Reduces risk that the proposed method only beats an artificially weak baseline.",
                safety_checks=["Preserve results.csv", "Preserve SUMMARY_JSON"],
            )
        if mutation.kind == "replication_check" and "REPLICATES = 10" in code:
            return PatchProposal(
                rationale="Increase repetitions to reduce noise in method comparison.",
                operations=[
                    PatchOperation(
                        find="REPLICATES = 10",
                        replace="REPLICATES = 20",
                        rationale="Run more repetitions for stability.",
                    )
                ],
                expected_effect="Improves robustness of aggregate metrics.",
                safety_checks=["Preserve results.csv", "Preserve SUMMARY_JSON", "Runtime remains small"],
            )
        if mutation.kind == "stress_test" and '"feasible": True,' in code:
            return PatchProposal(
                rationale="Mark stress-test rows explicitly so downstream analysis can distinguish normal and hard scenarios.",
                operations=[
                    PatchOperation(
                        find='"feasible": True,',
                        replace='"feasible": True,\n                    "stress_tested": True,',
                        rationale="Add an inspectable stress-test indicator column.",
                    )
                ],
                expected_effect="Makes stress-test outputs easier to filter without changing objective logic.",
                safety_checks=["Preserve results.csv", "Preserve SUMMARY_JSON", "Do not alter objective computation"],
            )
        return None
