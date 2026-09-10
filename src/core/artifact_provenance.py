"""Where each artifact's content actually came from.

The back half of this system was built so a number could not be trusted
without a check behind it. The front half had the opposite problem, and a worse
one: it did not fail, it *filled in*. The gap analysis named a research gap in
supply-chain chaos whatever the study was about. The algorithm plan proposed
dynamic programming over value density whatever the problem was. The reviewer
returned an overall score of 6 without reading the report. Every one of those
artifacts looked like a finding, and none of them had seen the run's inputs.

A confident wrong answer is worse than an absent one, because an absent one
tells you where to look. So each stage now declares one of three things:

* `llm` -- a model wrote it, from prompts this run actually sent;
* `derived` -- it was computed from this run's own inputs (the research goal,
  the domain profile, the literature index, the model draft);
* `not_generated` -- nothing wrote it, and the artifact says so in place of
  filler, naming what a real answer would have required.

The ledger collects those declarations and writes them to one file, so the
question "how much of this report is actually about my study?" has an answer
that does not require reading every artifact and recognising the boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


ContentSource = Literal["llm", "derived", "not_generated"]

#: The marker that makes a declared gap impossible to skim past, and easy to
#: grep for in a finished run directory.
GAP_MARKER = "NOT GENERATED"


class StageProvenance(BaseModel):
    """One stage's declaration about its own output."""

    stage: str
    source: ContentSource
    detail: str = ""
    #: Named inputs the stage actually read. Empty for `not_generated`.
    inputs_used: list[str] = Field(default_factory=list)
    #: What a real answer would have needed and this run did not have.
    missing: list[str] = Field(default_factory=list)

    @property
    def is_content(self) -> bool:
        return self.source != "not_generated"

    def one_line(self) -> str:
        if self.source == "not_generated":
            return f"`{self.stage}`: {GAP_MARKER} -- {self.detail}"
        inputs = ", ".join(f"`{item}`" for item in self.inputs_used) or "no named input"
        return f"`{self.stage}`: {self.source}, from {inputs}"


class ProvenanceLedger(BaseModel):
    entries: list[StageProvenance] = Field(default_factory=list)

    def record(
        self,
        stage: str,
        source: ContentSource,
        detail: str = "",
        inputs_used: list[str] | None = None,
        missing: list[str] | None = None,
    ) -> StageProvenance:
        entry = StageProvenance(
            stage=stage,
            source=source,
            detail=detail,
            inputs_used=list(inputs_used or []),
            missing=list(missing or []),
        )
        self.entries.append(entry)
        return entry

    @property
    def generated(self) -> list[StageProvenance]:
        return [entry for entry in self.entries if entry.is_content]

    @property
    def gaps(self) -> list[StageProvenance]:
        return [entry for entry in self.entries if not entry.is_content]

    def used_llm(self) -> bool:
        return any(entry.source == "llm" for entry in self.entries)

    def headline(self) -> str:
        """The one sentence a reader needs before trusting anything else."""

        total = len(self.entries)
        if not total:
            return "No stage declared where its content came from."
        generated, gaps = len(self.generated), len(self.gaps)
        llm = "some content was written by a language model" if self.used_llm() else (
            "no stage used a language model, so nothing here was reasoned about"
        )
        return (
            f"{generated} of {total} stage(s) produced content derived from this study's own"
            f" inputs; {gaps} declared a gap instead of filling one in. In this run, {llm}."
        )

    def to_markdown(self) -> str:
        lines = ["# Artifact Provenance", "", self.headline(), ""]
        if not self.entries:
            return "\n".join(lines) + "\n"
        lines.append("## Stages")
        lines.append("")
        lines.append("| stage | source | inputs used |")
        lines.append("| --- | --- | --- |")
        for entry in self.entries:
            inputs = ", ".join(f"`{item}`" for item in entry.inputs_used) or "--"
            lines.append(f"| {entry.stage} | {entry.source} | {inputs} |")
        lines.append("")
        if self.gaps:
            lines.append("## Declared Gaps")
            lines.append("")
            lines.append(
                "These stages produced no content. That is deliberate: filler that looks like a"
                " finding is harder to notice than an admitted gap."
            )
            lines.append("")
            for entry in self.gaps:
                lines.append(f"### {entry.stage}")
                lines.append(f"- {entry.detail}")
                if entry.missing:
                    lines.append("- a real answer would need:")
                    lines.extend(f"  - {item}" for item in entry.missing)
                lines.append("")
        lines.append("## How To Read This")
        lines.append("")
        lines.extend(
            [
                "- `llm`: a model wrote it from a prompt this run actually sent. It still needs checking.",
                "- `derived`: computed from the run's own inputs by explicit rules you can read in the source.",
                f"- `not_generated`: nothing wrote it. The artifact carries a `{GAP_MARKER}` block instead of filler.",
                "",
                "A stage marked `derived` is not thereby correct. It only means the output is a"
                " function of this study's inputs rather than a constant.",
            ]
        )
        return "\n".join(lines) + "\n"


def declared_gap(what: str, why: str, needs: list[str], next_step: str = "") -> str:
    """A markdown block that admits an absence instead of filling it.

    Deliberately loud. The failure mode being designed against is a reader
    skimming a well-formatted section and taking it for a result.
    """

    lines = [
        f"> **{GAP_MARKER}: {what}**",
        ">",
        f"> {why}",
    ]
    if needs:
        lines.append(">")
        lines.append("> A real answer here would require:")
        lines.extend(f"> - {item}" for item in needs)
    if next_step:
        lines.append(">")
        lines.append(f"> {next_step}")
    return "\n".join(lines)


def write_provenance(run_dir: str | Path, ledger: ProvenanceLedger) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "artifact_provenance.json").write_text(
        json.dumps(ledger.model_dump(), indent=2), encoding="utf-8"
    )
    path = root / "artifact_provenance.md"
    path.write_text(ledger.to_markdown(), encoding="utf-8")
    return path
