from __future__ import annotations

import re
import hashlib
import math
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class ClaimGroundingRecord(BaseModel):
    claim: str
    best_source: str | None = None
    best_score: float = 0.0
    lexical_score: float = 0.0
    embedding_score: float = 0.0
    evidence_snippet: str | None = None
    status: str


class LiteratureGroundingReport(BaseModel):
    claims_checked: int
    grounded_claims: int
    weak_claims: int
    records: list[ClaimGroundingRecord] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Literature Grounding",
            "",
            f"- claims_checked: {self.claims_checked}",
            f"- grounded_claims: {self.grounded_claims}",
            f"- weak_claims: {self.weak_claims}",
            "",
            "## Claim Evidence Matrix",
        ]
        if not self.records:
            lines.append("- No candidate claims were extracted.")
        for record in self.records:
            lines.append(f"### {record.status}: {record.claim}")
            lines.append(f"- best_source: `{record.best_source or 'none'}`")
            lines.append(f"- best_score: {record.best_score:.3f}")
            lines.append(f"- lexical_score: {record.lexical_score:.3f}")
            lines.append(f"- embedding_score: {record.embedding_score:.3f}")
            if record.evidence_snippet:
                lines.append(f"- evidence_snippet: {record.evidence_snippet[:400]}")
            lines.append("")
        lines.append("## Human Verification")
        lines.append("- Grounded means keyword overlap with local text, not citation correctness.")
        lines.append("- Verify authors, years, venues, equations, and contribution claims manually.")
        return "\n".join(lines) + "\n"


def build_literature_grounding(
    report_path: str | Path,
    literature_index_path: str | Path | None,
    output_dir: str | Path,
) -> LiteratureGroundingReport:
    output = Path(output_dir)
    report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore") if Path(report_path).exists() else ""
    claims = _extract_candidate_claims(report_text)
    literature = _load_literature(literature_index_path)
    records: list[ClaimGroundingRecord] = []
    for claim in claims:
        best = _best_source(claim, literature)
        status = "grounded" if best and best["best_score"] >= 0.18 else "weak"
        records.append(
            ClaimGroundingRecord(
                claim=claim,
                best_source=best["record"]["filename"] if best else None,
                best_score=best["best_score"] if best else 0.0,
                lexical_score=best["lexical_score"] if best else 0.0,
                embedding_score=best["embedding_score"] if best else 0.0,
                evidence_snippet=best["record"].get("snippet") if best else None,
                status=status,
            )
        )
    grounded = sum(1 for record in records if record.status == "grounded")
    result = LiteratureGroundingReport(
        claims_checked=len(records),
        grounded_claims=grounded,
        weak_claims=len(records) - grounded,
        records=records,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "literature_grounding.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (output / "literature_grounding.md").write_text(result.to_markdown(), encoding="utf-8")
    return result


def _extract_candidate_claims(text: str) -> list[str]:
    plain = re.sub(r"[#*_`>-]", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", plain)
    claim_markers = [
        "gap",
        "novel",
        "new",
        "improve",
        "outperform",
        "contribution",
        "literature",
        "shows",
        "suggests",
        "reliable",
        "robust",
    ]
    claims = []
    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if 40 <= len(sentence) <= 280 and any(marker in sentence.lower() for marker in claim_markers):
            claims.append(sentence)
    return claims[:20]


def _load_literature(literature_index_path: str | Path | None) -> list[dict]:
    if not literature_index_path:
        return []
    path = Path(literature_index_path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.fillna("").to_dict("records")


def _best_source(claim: str, literature: list[dict]) -> dict | None:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return None
    claim_vector = _hashed_embedding(claim)
    best_record = None
    best_payload = None
    for record in literature:
        haystack = " ".join(str(record.get(key, "")) for key in ["title_guess", "keyword_hits", "snippet", "category"])
        source_tokens = _tokens(haystack)
        if not source_tokens:
            continue
        overlap = len(claim_tokens & source_tokens)
        lexical_score = overlap / max(len(claim_tokens), 1)
        embedding_score = _cosine(claim_vector, _hashed_embedding(haystack))
        score = 0.55 * lexical_score + 0.45 * embedding_score
        if best_payload is None or score > best_payload["best_score"]:
            best_record = record
            best_payload = {
                "record": best_record,
                "best_score": score,
                "lexical_score": lexical_score,
                "embedding_score": embedding_score,
            }
    return best_payload


def _tokens(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "that",
        "with",
        "this",
        "from",
        "into",
        "have",
        "will",
        "were",
        "been",
        "for",
        "are",
        "not",
    }
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()) if token not in stopwords}


def _hashed_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right)))
