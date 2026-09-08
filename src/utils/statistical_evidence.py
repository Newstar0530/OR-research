"""Method-level statistics with an honest significance verdict.

The previous version of this module could label a comparison
`strong_preliminary` purely because the sample was large and the point estimate
pointed the right way -- even when the p-value was 0.76. That is worse than no
label at all, so the significance test now **vetoes** the optimistic labels: an
observed improvement that is not distinguishable from noise is reported as
`inconclusive_not_significant`, not as strong evidence.

Welch's t-test is used when SciPy is available (correct for small, unequal-
variance samples, which is the normal case in OR experiments). Without SciPy it
falls back to a normal approximation, and says so in `test_used`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

try:  # SciPy is an optional extra; the module must work without it.
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - depends on runtime extras
    _scipy_stats = None


DEFAULT_SIGNIFICANCE_LEVEL = 0.05
MIN_N_FOR_POWER = 5


class MethodStatistic(BaseModel):
    method: str
    n: int
    mean: float
    std: float
    sem: float = 0.0
    ci95_low: float
    ci95_high: float


class StatisticalEvidenceReport(BaseModel):
    metric: str
    objective_direction: str
    method_statistics: list[MethodStatistic] = Field(default_factory=list)
    best_method: str | None = None
    baseline_method: str | None = None
    proposed_method: str | None = None
    best_vs_baseline_effect: float | None = None
    standardized_effect_size: float | None = None
    welch_z_statistic: float | None = None
    approximate_p_value: float | None = None
    degrees_of_freedom: float | None = None
    test_used: str = "not_applicable"
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL
    is_statistically_significant: bool | None = None
    evidence_strength: str
    warnings: list[str] = Field(default_factory=list)

    @property
    def supports_improvement_claim(self) -> bool:
        """The only combination that licenses "the proposed method is better"."""

        return bool(
            self.is_statistically_significant
            and self.best_vs_baseline_effect is not None
            and self.best_vs_baseline_effect > 0
        )

    def to_markdown(self) -> str:
        def fmt(value: float | None) -> str:
            return "unknown" if value is None else f"{value:.6g}"

        significance = (
            "not tested"
            if self.is_statistically_significant is None
            else ("yes" if self.is_statistically_significant else "no")
        )
        lines = [
            "# Statistical Evidence",
            "",
            f"- metric: {self.metric}",
            f"- objective_direction: {self.objective_direction}",
            f"- evidence_strength: {self.evidence_strength}",
            f"- statistically_significant (alpha={self.significance_level:g}): {significance}",
            f"- supports_improvement_claim: {self.supports_improvement_claim}",
            "",
            "## Method Statistics",
            "| method | n | mean | std | sem | ci95_low | ci95_high |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for stat in self.method_statistics:
            lines.append(
                f"| {stat.method} | {stat.n} | {stat.mean:.6g} | {stat.std:.6g} | {stat.sem:.6g} | "
                f"{stat.ci95_low:.6g} | {stat.ci95_high:.6g} |"
            )
        lines.extend(
            [
                "",
                "## Proposed vs Baseline",
                f"- baseline_method: `{self.baseline_method or 'unknown'}`",
                f"- proposed_method: `{self.proposed_method or 'unknown'}`",
                f"- best_method (by mean): `{self.best_method or 'unknown'}`",
                f"- effect (positive = proposed better): {fmt(self.best_vs_baseline_effect)}",
                f"- standardized_effect_size (Cohen's d): {fmt(self.standardized_effect_size)}",
                f"- test_used: {self.test_used}",
                f"- test_statistic: {fmt(self.welch_z_statistic)}",
                f"- degrees_of_freedom: {fmt(self.degrees_of_freedom)}",
                f"- p_value: {fmt(self.approximate_p_value)}",
                "",
                "## Warnings",
            ]
        )
        lines.extend(f"- {item}" for item in self.warnings or ["None."])
        lines.extend(
            [
                "",
                "## Human Verification",
                "- A point-estimate improvement is not a result. Only "
                "`supports_improvement_claim: True` licenses a comparative claim, and even then the "
                "test assumes independent observations.",
                "- Paired designs (same instances solved by both methods) are more powerful than the "
                "unpaired test used here; consider a paired test before publication.",
            ]
        )
        return "\n".join(lines) + "\n"


def _t_critical(df: float) -> float:
    if _scipy_stats is None or df <= 0:
        return 1.96
    try:
        return float(_scipy_stats.t.ppf(0.975, df))
    except Exception:  # pragma: no cover - defensive
        return 1.96


def _welch_test(
    proposed: MethodStatistic, baseline: MethodStatistic, maximize: bool
) -> tuple[float | None, float | None, float | None, str]:
    """Returns (statistic, p_value, degrees_of_freedom, test_used)."""

    if proposed.n < 2 or baseline.n < 2:
        return None, None, None, "not_applicable_n_too_small"
    variance_term = (proposed.std**2 / proposed.n) + (baseline.std**2 / baseline.n)
    standard_error = math.sqrt(variance_term)
    if standard_error == 0:
        return None, None, None, "not_applicable_zero_variance"
    difference = (
        proposed.mean - baseline.mean if maximize else baseline.mean - proposed.mean
    )
    statistic = difference / standard_error
    numerator = variance_term**2
    denominator = (proposed.std**4 / (proposed.n**2 * (proposed.n - 1))) + (
        baseline.std**4 / (baseline.n**2 * (baseline.n - 1))
    )
    degrees_of_freedom = numerator / denominator if denominator > 0 else None
    if _scipy_stats is not None and degrees_of_freedom:
        try:
            p_value = float(2.0 * _scipy_stats.t.sf(abs(statistic), degrees_of_freedom))
            return statistic, p_value, float(degrees_of_freedom), "welch_t"
        except Exception:  # pragma: no cover - defensive
            pass
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, p_value, degrees_of_freedom, "normal_approximation"


def _cohens_d(
    proposed: MethodStatistic, baseline: MethodStatistic, effect: float
) -> float | None:
    total = proposed.n + baseline.n
    if total <= 2:
        return None
    pooled_variance = (
        (proposed.n - 1) * proposed.std**2 + (baseline.n - 1) * baseline.std**2
    ) / (total - 2)
    pooled_sd = math.sqrt(pooled_variance)
    if pooled_sd <= 0:
        return None
    return effect / pooled_sd


def _pick_baseline(stats: list[MethodStatistic], requested: str | None) -> MethodStatistic:
    if requested:
        for stat in stats:
            if stat.method == requested:
                return stat
    for stat in stats:
        lowered = stat.method.lower()
        if "base" in lowered or "greedy" in lowered:
            return stat
    return stats[0]


def _pick_proposed(
    stats: list[MethodStatistic],
    requested: str | None,
    baseline: MethodStatistic,
    maximize: bool,
) -> MethodStatistic:
    if requested:
        for stat in stats:
            if stat.method == requested:
                return stat
    candidates = [stat for stat in stats if stat.method != baseline.method] or stats
    return (
        max(candidates, key=lambda item: item.mean)
        if maximize
        else min(candidates, key=lambda item: item.mean)
    )


def _classify(
    n_methods: int,
    effect: float | None,
    significant: bool | None,
    min_n: int,
) -> str:
    """Evidence label. A failed significance test always vetoes an optimistic label."""

    if n_methods < 2:
        return "single_method_no_comparison"
    if effect is None:
        return "preliminary"
    if effect <= 0:
        return "no_improvement_observed"
    if significant is False:
        return "inconclusive_not_significant"
    if significant is True:
        return "statistically_promising" if min_n >= MIN_N_FOR_POWER else "significant_but_underpowered"
    if min_n >= 10:
        return "strong_preliminary_untested"
    if min_n >= MIN_N_FOR_POWER:
        return "moderate_untested"
    return "preliminary"


def evaluate_statistical_evidence(
    results_csv: str | Path,
    output_dir: str | Path,
    metric: str = "objective",
    objective_direction: str = "minimize",
    baseline_method: str | None = None,
    proposed_method: str | None = None,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> StatisticalEvidenceReport:
    path = Path(results_csv)
    warnings: list[str] = []
    if not path.exists():
        return _write(
            StatisticalEvidenceReport(
                metric=metric,
                objective_direction=objective_direction,
                evidence_strength="missing_results",
                warnings=["results.csv was not found."],
                significance_level=significance_level,
            ),
            output_dir,
        )

    df = pd.read_csv(path)
    if "method" not in df.columns or metric not in df.columns:
        return _write(
            StatisticalEvidenceReport(
                metric=metric,
                objective_direction=objective_direction,
                evidence_strength="insufficient_columns",
                warnings=[f"Need method and {metric} columns."],
                significance_level=significance_level,
            ),
            output_dir,
        )

    # Only verified rows may contribute to a statistical claim.
    if "verification_ok" in df.columns:
        broken = len(df) - int(df["verification_ok"].astype(str).str.lower().isin({"true", "1"}).sum())
        if broken:
            warnings.append(
                f"Excluded {broken} row(s) that failed independent solution verification."
            )
            df = df[df["verification_ok"].astype(str).str.lower().isin({"true", "1"})]
    if "feasible" in df.columns:
        infeasible = len(df) - int(df["feasible"].astype(str).str.lower().isin({"true", "1"}).sum())
        if infeasible:
            warnings.append(f"Excluded {infeasible} row(s) whose solution was not feasible.")
            df = df[df["feasible"].astype(str).str.lower().isin({"true", "1"})]

    stats: list[MethodStatistic] = []
    for method, group in df.groupby("method"):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        n = int(values.shape[0])
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0
        half_width = _t_critical(n - 1) * sem if n > 1 else 0.0
        stats.append(
            MethodStatistic(
                method=str(method),
                n=n,
                mean=mean,
                std=std,
                sem=sem,
                ci95_low=mean - half_width,
                ci95_high=mean + half_width,
            )
        )
        if n < 3:
            warnings.append(f"Method `{method}` has fewer than 3 observations.")

    if not stats:
        return _write(
            StatisticalEvidenceReport(
                metric=metric,
                objective_direction=objective_direction,
                evidence_strength="no_numeric_data",
                warnings=warnings or ["No numeric metric data."],
                significance_level=significance_level,
            ),
            output_dir,
        )

    maximize = objective_direction == "maximize"
    best = max(stats, key=lambda item: item.mean) if maximize else min(stats, key=lambda item: item.mean)
    baseline = _pick_baseline(stats, baseline_method)
    proposed = _pick_proposed(stats, proposed_method, baseline, maximize)

    effect: float | None = None
    statistic = p_value = degrees_of_freedom = None
    test_used = "not_applicable"
    standardized: float | None = None
    significant: bool | None = None

    if proposed.method == baseline.method:
        warnings.append(
            "Baseline and proposed method resolved to the same method; no comparison was made. "
            "Set `baseline_method` and `proposed_method` in the config."
        )
    else:
        effect = (
            proposed.mean - baseline.mean if maximize else baseline.mean - proposed.mean
        )
        statistic, p_value, degrees_of_freedom, test_used = _welch_test(proposed, baseline, maximize)
        standardized = _cohens_d(proposed, baseline, effect)
        if p_value is not None:
            significant = bool(p_value < significance_level)
            if not significant and effect > 0:
                warnings.append(
                    f"Observed effect {effect:.6g} in favour of `{proposed.method}` is NOT "
                    f"statistically significant (p={p_value:.4g} >= {significance_level:g}); it is not "
                    "distinguishable from noise at this sample size."
                )
            if significant and effect <= 0:
                warnings.append(
                    f"`{baseline.method}` is significantly better than `{proposed.method}` "
                    f"(p={p_value:.4g}). The hypothesis is contradicted, not merely unsupported."
                )
        else:
            warnings.append(
                f"No significance test could be run ({test_used}); the effect is a point estimate only."
            )

    min_n = min(item.n for item in stats)
    if min_n < MIN_N_FOR_POWER:
        warnings.append(
            f"Smallest group has n={min_n} (< {MIN_N_FOR_POWER}); the test is underpowered. "
            "Add instance seeds or replications."
        )
    if len(stats) < 2:
        warnings.append("Only one method was available; comparative evidence is weak.")

    report = StatisticalEvidenceReport(
        metric=metric,
        objective_direction=objective_direction,
        method_statistics=stats,
        best_method=best.method,
        baseline_method=baseline.method,
        proposed_method=proposed.method,
        best_vs_baseline_effect=effect,
        standardized_effect_size=standardized,
        welch_z_statistic=statistic,
        approximate_p_value=p_value,
        degrees_of_freedom=degrees_of_freedom,
        test_used=test_used,
        significance_level=significance_level,
        is_statistically_significant=significant,
        evidence_strength=_classify(len(stats), effect, significant, min_n),
        warnings=warnings,
    )
    return _write(report, output_dir)


def _write(report: StatisticalEvidenceReport, output_dir: str | Path) -> StatisticalEvidenceReport:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "statistical_evidence.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (root / "statistical_evidence.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
