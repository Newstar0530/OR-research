from __future__ import annotations

import re

from src.core.mutation import MutationSpec
from src.llm_client import LLMClient


class ExperimentRefinerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, base_code: str, plan: str, iteration: int, branch_index: int, mutation: MutationSpec | None = None) -> str:
        """Create a bounded code variant for autonomous search.

        The MVP keeps edits mechanical and inspectable. It varies seeds and, for the
        generic scaffold, changes the proposed-method effect size.
        """

        code = base_code
        seed = 42 + iteration * 100 + branch_index
        code = code.replace("default_rng(42)", f"default_rng({seed})")
        code = code.replace("random.Random(42)", f"random.Random({seed})")
        code = code.replace('"seed": 42', f'"seed": {seed}')
        if mutation:
            for source, target in mutation.replacements.items():
                code = code.replace(source, target)
        effect_range = self._extract_effect_range(plan)
        if effect_range:
            code = code.replace("PROPOSED_EFFECT_LOW = 0.5", f"PROPOSED_EFFECT_LOW = {effect_range[0]}")
            code = code.replace("PROPOSED_EFFECT_HIGH = 3.0", f"PROPOSED_EFFECT_HIGH = {effect_range[1]}")
            code = code.replace("rng.uniform(0.5, 3.0)", f"rng.uniform({effect_range[0]}, {effect_range[1]})")
        elif branch_index == 1:
            code = code.replace("PROPOSED_EFFECT_LOW = 0.5", "PROPOSED_EFFECT_LOW = 1.0")
            code = code.replace("PROPOSED_EFFECT_HIGH = 3.0", "PROPOSED_EFFECT_HIGH = 4.5")
            code = code.replace("rng.uniform(0.5, 3.0)", "rng.uniform(1.0, 4.5)")
        elif branch_index == 2:
            code = code.replace("PROPOSED_EFFECT_LOW = 0.5", "PROPOSED_EFFECT_LOW = 0.1")
            code = code.replace("PROPOSED_EFFECT_HIGH = 3.0", "PROPOSED_EFFECT_HIGH = 2.0")
            code = code.replace("rng.uniform(0.5, 3.0)", "rng.uniform(0.1, 2.0)")
        mutation_text = mutation.to_plan_suffix() if mutation else "mutation=none"
        header = f'"""Autonomous variant: iteration={iteration}, branch={branch_index}, plan={plan}; {mutation_text}."""'
        return _insert_header_after_future_imports(code, header)

    @staticmethod
    def _extract_effect_range(plan: str) -> tuple[float, float] | None:
        match = re.search(r"effect_range\s*=\s*([0-9.]+)\s*,\s*([0-9.]+)", plan)
        if not match:
            return None
        low = float(match.group(1))
        high = float(match.group(2))
        if low >= high:
            return None
        return low, high


def _insert_header_after_future_imports(code: str, header: str) -> str:
    lines = code.splitlines()
    insert_at = 0
    while insert_at < len(lines) and (not lines[insert_at].strip() or lines[insert_at].lstrip().startswith("#")):
        insert_at += 1
    while insert_at < len(lines) and lines[insert_at].startswith('"""'):
        if lines[insert_at].count('"""') >= 2:
            insert_at += 1
            while insert_at < len(lines) and not lines[insert_at].strip():
                insert_at += 1
            continue
        insert_at += 1
        while insert_at < len(lines) and '"""' not in lines[insert_at]:
            insert_at += 1
        if insert_at < len(lines):
            insert_at += 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
    while insert_at < len(lines) and lines[insert_at].startswith("from __future__ import"):
        insert_at += 1
    updated = lines[:insert_at] + [header] + lines[insert_at:]
    return "\n".join(updated) + ("\n" if code.endswith("\n") else "")
