"""Produce a variant of an experiment, and report whether it is really a variant.

The bounded find/replace this started as is fine when it applies. The problem
was what happened when it did not: `code.replace(source, target)` on a template
that does not contain `PROPOSED_EFFECT_LOW = 0.5` silently returns the same
string. Every replacement missed, the only thing that changed was the docstring
header the refiner inserts, and the node became a copy of its parent -- which
the search then scored and ranked as an explored branch. Nothing noticed,
because the refiner returned a bare string and no caller compared it to what it
was given.

So this now returns a `MutationOutcome` that says which replacement keys were
present, which were absent, and whether the behaviour changed at all. And when
a model is available it is asked to apply the mutation's *intent* to the whole
script, which works on templates the replacement keys know nothing about. That
rewrite goes through the same guard as a repair rewrite, plus one more check: a
variant identical to its parent is rejected, because re-measuring the parent is
not exploration.
"""

from __future__ import annotations

import re

from src.core.generated_code_guard import behavioural_lines, extract_code, validate_mutation
from src.core.mutation import MutationOutcome, MutationSpec
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError


MUTATION_SYSTEM_PROMPT = (
    "You modify a working Operations Research experiment script to explore one specific "
    "variation. You change the smallest set of things that realises the requested variation, "
    "you keep the script runnable and self-contained, and you preserve its outputs. You never "
    "rewrite the experiment from scratch and you never weaken its measurements to make a result "
    "look better."
)

#: Attempts per mutation. A rejection is fed back, so this is a conversation.
MAX_MUTATION_ATTEMPTS = 2


class ExperimentRefinerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        base_code: str,
        plan: str,
        iteration: int,
        branch_index: int,
        mutation: MutationSpec | None = None,
    ) -> MutationOutcome:
        """Create a variant. Returns the code and what was actually done to it."""

        seed = 42 + iteration * 100 + branch_index
        header = self._header(plan, iteration, branch_index, mutation)

        if mutation is not None and not self.llm.use_mock:
            outcome = self._llm_variant(base_code, plan, mutation, seed, header)
            if outcome is not None:
                return outcome

        return self._string_variant(base_code, plan, mutation, seed, header)

    # -- the bounded, deterministic path -----------------------------------

    def _string_variant(
        self,
        base_code: str,
        plan: str,
        mutation: MutationSpec | None,
        seed: int,
        header: str,
    ) -> MutationOutcome:
        code = base_code
        matched: list[str] = []
        unmatched: list[str] = []

        for source, target in (
            ("default_rng(42)", f"default_rng({seed})"),
            ("random.Random(42)", f"random.Random({seed})"),
            ('"seed": 42', f'"seed": {seed}'),
        ):
            if source in code:
                code = code.replace(source, target)
                matched.append(source)

        if mutation:
            for source, target in mutation.replacements.items():
                if source in code:
                    code = code.replace(source, target)
                    matched.append(source)
                else:
                    unmatched.append(source)

        for source, target in self._effect_replacements(plan, self._extract_effect_range(plan)):
            if source in code:
                code = code.replace(source, target)
                matched.append(source)

        code = _insert_header_after_future_imports(code, header)
        notes: list[str] = []
        if unmatched:
            notes.append(
                f"{len(unmatched)} of {len(unmatched) + len([m for m in matched if mutation and m in mutation.replacements])} "
                "mutation replacement(s) matched nothing in this template: "
                + ", ".join(f"`{item}`" for item in unmatched[:6])
            )
        # Only one question applies to this path: did anything change? The
        # replacements are bounded and the parent already ran, so the rest of
        # the generated-code guard is the rewrite path's concern -- and the
        # experiment contract checks the outputs after execution regardless.
        if behavioural_lines(code) == behavioural_lines(base_code):
            notes.append(
                "The variant is byte-identical to its parent apart from the header, so this node "
                "would re-measure the parent. Exact string replacement only works on templates "
                "that expose these constant names; a language model can apply the mutation's "
                "intent to any template."
            )
            return MutationOutcome(
                code=code,
                method="none",
                applied=False,
                matched_replacements=matched,
                unmatched_replacements=unmatched,
                notes=notes,
            )
        return MutationOutcome(
            code=code,
            method="string_replacement",
            applied=True,
            matched_replacements=matched,
            unmatched_replacements=unmatched,
            notes=notes,
        )

    @staticmethod
    def _effect_replacements(
        plan: str, effect_range: tuple[float, float] | None
    ) -> list[tuple[str, str]]:
        """The generic template's effect knobs, when the plan names a range.

        These only exist on `generic_or_template.py`, which is why they are
        listed as replacements rather than assumed: on any other template they
        match nothing, and the outcome now says so instead of pretending.
        """

        if not effect_range:
            return []
        low, high = effect_range
        return [
            ("PROPOSED_EFFECT_LOW = 0.5", f"PROPOSED_EFFECT_LOW = {low}"),
            ("PROPOSED_EFFECT_HIGH = 3.0", f"PROPOSED_EFFECT_HIGH = {high}"),
            ("rng.uniform(0.5, 3.0)", f"rng.uniform({low}, {high})"),
        ]

    # -- the model path ----------------------------------------------------

    def _llm_variant(
        self,
        base_code: str,
        plan: str,
        mutation: MutationSpec,
        seed: int,
        header: str,
    ) -> MutationOutcome | None:
        """Apply the mutation's intent to the whole script, then check the result."""

        notes: list[str] = []
        feedback = ""
        for attempt in range(MAX_MUTATION_ATTEMPTS):
            prompt = self._prompt(base_code, plan, mutation, seed, feedback)
            try:
                response = self.llm.chat(MUTATION_SYSTEM_PROMPT, prompt, stage="experiment_mutation")
            except Exception as exc:
                if isinstance(exc, LLMCallError) and exc.is_permanent:
                    raise
                reason = exc.summary() if isinstance(exc, LLMCallError) else str(exc)
                notes.append(f"Attempt {attempt + 1}: the model call failed ({reason}).")
                return None
            candidate = extract_code(response)
            verdict = validate_mutation(candidate, base_code)
            if verdict.ok:
                notes.extend(verdict.notes)
                return MutationOutcome(
                    code=_insert_header_after_future_imports(candidate, header),
                    method="llm",
                    applied=True,
                    notes=notes,
                )
            notes.append(f"Attempt {attempt + 1} rejected: " + "; ".join(verdict.violations))
            feedback = verdict.to_feedback()
        # Falling through to the string path is better than shipping a rejected
        # rewrite, and the notes travel with the outcome.
        return None

    @staticmethod
    def _prompt(
        base_code: str, plan: str, mutation: MutationSpec, seed: int, feedback: str = ""
    ) -> str:
        sections = [
            "Modify the experiment below to explore one variation.",
            "",
            f"Variation to apply ({mutation.kind}): {mutation.description}",
            f"Branch plan: {plan}",
            f"Use {seed} as the random seed so this variant is reproducible and distinct.",
        ]
        if mutation.replacements:
            sections.append("")
            sections.append(
                "On the reference template this variation is these edits. Apply the equivalent "
                "change to whatever this script actually uses, even if these exact names do not "
                "appear in it:"
            )
            sections.extend(
                f"- `{source}` -> `{target}`" for source, target in mutation.replacements.items()
            )
        sections.extend(
            [
                "",
                "Rules:",
                "- Return the complete modified script and nothing else.",
                "- Keep writing `results.csv` and keep printing the final `SUMMARY_JSON:` line.",
                "- Do not import socket, urllib, requests, subprocess, multiprocessing or shutil.",
                "- Do not call eval, exec or any shell.",
                "- Write only inside the script's own directory.",
                "- Change what the experiment does, not only its comments.",
                "- Do not make the proposed method look better by weakening the baseline or the "
                "verification. A mutation that improves the result by measuring less is a defect.",
                "",
                "Current script:",
                "```python",
                base_code,
                "```",
            ]
        )
        if feedback:
            sections.extend(["", feedback])
        return "\n".join(sections)

    # -- shared ------------------------------------------------------------

    @staticmethod
    def _header(plan: str, iteration: int, branch_index: int, mutation: MutationSpec | None) -> str:
        mutation_text = mutation.to_plan_suffix() if mutation else "mutation=none"
        return f'"""Autonomous variant: iteration={iteration}, branch={branch_index}, plan={plan}; {mutation_text}."""'

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
