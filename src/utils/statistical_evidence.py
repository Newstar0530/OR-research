"""Method-level statistics with an honest significance verdict.

Two defects have been fixed here, and both were of the same kind: the report
sounded more certain than the numbers allowed.

The first was labelling. A comparison could be called `strong_preliminary`
purely because the sample was large and the point estimate pointed the right
way -- even at p=0.76. The significance test now **vetoes** the optimistic
labels: an observed improvement that is not distinguishable from noise is
reported as `inconclusive_not_significant`.

The second was the test itself. OR experiments almost always solve *the same
instances* with every method, which makes the observations paired. Feeding
paired data to an unpaired test is not a matter of taste: it discards the
instance-to-instance variation that both methods share -- usually the largest
source of spread in a benchmark -- and it assumes an independence the design
does not have. So the design is now detected from the data, one test is chosen
from it, and that test is the one reported. There is no second p-value on offer
to pick from afterwards.

Detection is a check, not an assumption. Pairing is used only when the identity
columns line up one-to-one across the two methods; otherwise the module falls
back to Welch's unpaired t-test and says why in `warnings`.

Welch's t-test (unpaired) and Student's paired t-test both use SciPy when it is
available. Without SciPy the p-value falls back to a normal approximation and
`test_used` says so. When SciPy is present a Wilcoxon signed-rank test runs
alongside the paired t-test as a robustness check -- it can raise doubt about a
verdict but never upgrade one.
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
    test_statistic: float | None = None
    approximate_p_value: float | None = None
    degrees_of_freedom: float | None = None
    test_used: str = "not_applicable"
    #: `paired` when the same instances were solved by both methods and the
    #: identity columns line up one-to-one, `unpaired` otherwise. Chosen before
    #: any test is run, so the design decides the test rather than the reverse.
    design: str = "not_applicable"
    #: The columns that identified an instance. Empty when pairing was refused.
    pairing_key: list[str] = Field(default_factory=list)
    n_pairs: int | None = None
    #: Per-instance outcomes. `ties` matters: a benchmark where both methods hit
    #: the proved optimum on most instances can look significant off a handful
    #: of non-tied pairs.
    n_wins: int | None = None
    n_losses: int | None = None
    n_ties: int | None = None
    paired_mean_difference: float | None = None
    paired_sd_difference: float | None = None
    #: Distribution-free robustness check. It can cast doubt on the t-test's
    #: verdict but never overturn or upgrade it -- see `warnings`.
    wilcoxon_p_value: float | None = None
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

        effect_label = "Cohen's d_z, paired" if self.design == "paired" else "Cohen's d"
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
                f"- standardized_effect_size ({effect_label}): {fmt(self.standardized_effect_size)}",
                f"- design: {self.design}",
                f"- test_used: {self.test_used}",
                f"- test_statistic: {fmt(self.test_statistic)}",
                f"- degrees_of_freedom: {fmt(self.degrees_of_freedom)}",
                f"- p_value: {fmt(self.approximate_p_value)}",
            ]
        )
        if self.design == "paired":
            lines.extend(
                [
                    "",
                    "### Paired Design",
                    f"- pairing_key: {', '.join(f'`{key}`' for key in self.pairing_key) or 'none'}",
                    f"- n_pairs: {self.n_pairs if self.n_pairs is not None else 'unknown'}",
                    f"- per-instance outcome: {self.n_wins} win / {self.n_losses} loss / {self.n_ties} tie",
                    f"- mean paired difference (positive = proposed better): {fmt(self.paired_mean_difference)}",
                    f"- sd of paired differences: {fmt(self.paired_sd_difference)}",
                    f"- wilcoxon_p_value (robustness check only): {fmt(self.wilcoxon_p_value)}",
                ]
            )
        lines.extend(
            [
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
                "`supports_improvement_claim: True` licenses a comparative claim.",
                "- The design above was detected from the data and fixed the test before it was "
                "run. If `design` is `unpaired`, the warnings say why pairing was refused; the fix "
                "is to make the experiment solve the same instances with both methods, not to "
                "re-run the analysis a different way.",
                "- Statistical significance is not practical significance. Check the effect size "
                "against what would matter for the application.",
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


#: Columns that could identify "the same problem instance", most specific first.
IDENTITY_COLUMNS: tuple[str, ...] = ("instance_id", "instance", "seed", "replication")


class PairedSample(BaseModel):
    """Per-instance differences, oriented so positive always means better."""

    key: list[str]
    differences: list[float]
    dropped_keys: int = 0

    @property
    def n(self) -> int:
        return len(self.differences)


def detect_paired_sample(
    df: pd.DataFrame,
    metric: str,
    baseline_method: str,
    proposed_method: str,
    maximize: bool,
) -> tuple[PairedSample | None, list[str]]:
    """Pair the two methods' rows by instance identity, or explain the refusal.

    Pairing is a claim about how the experiment was run, so it is checked rather
    than assumed. Every identity value must appear exactly once per method; a
    method that solved one instance twice under the same key makes the pairing
    ambiguous, and an ambiguous pairing is worse than none because it silently
    invents a design the data does not have.
    """

    notes: list[str] = []
    key = [column for column in IDENTITY_COLUMNS if column in df.columns]
    if not key:
        notes.append(
            "No instance identity column (`instance_id`, `instance`, `seed`, `replication`) is "
            "present, so the observations cannot be paired. Falling back to the unpaired test."
        )
        return None, notes
    # `instance_id` alone does not identify a run when several seeds were used,
    # and `seed` alone identifies one only because the generators derive the
    # instance from it. Both are kept when both exist.
    if "instance_id" in key or "instance" in key:
        key = [column for column in key if column != "replication"]
    else:
        notes.append(
            f"No instance column; pairing was inferred from `{'`, `'.join(key)}` alone. This is "
            "correct only if the same seed produces the same instance for every method."
        )

    frames = {}
    for method in (baseline_method, proposed_method):
        subset = df[df["method"].astype(str) == method][key + [metric]].copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset.dropna(subset=[metric])
        duplicated = int(subset.duplicated(subset=key).sum())
        if duplicated:
            notes.append(
                f"Method `{method}` has {duplicated} row(s) sharing an identity key "
                f"({', '.join(key)}), so the pairing would be ambiguous. Falling back to the "
                "unpaired test."
            )
            return None, notes
        frames[method] = subset.set_index(key)[metric]

    baseline_values, proposed_values = frames[baseline_method], frames[proposed_method]
    shared = baseline_values.index.intersection(proposed_values.index)
    dropped = (len(baseline_values) - len(shared)) + (len(proposed_values) - len(shared))
    if len(shared) < 2:
        notes.append(
            f"Only {len(shared)} instance(s) were solved by both methods, which is not enough to "
            "pair. Falling back to the unpaired test."
        )
        return None, notes
    if dropped:
        notes.append(
            f"{dropped} row(s) had no counterpart in the other method and were excluded from the "
            "paired comparison. An unbalanced benchmark can bias a comparison if the missing "
            "instances are the hard ones."
        )

    left = proposed_values.loc[shared].astype(float)
    right = baseline_values.loc[shared].astype(float)
    differences = (left - right) if maximize else (right - left)
    return (
        PairedSample(key=key, differences=[float(value) for value in differences], dropped_keys=dropped),
        notes,
    )


def _paired_test(
    sample: PairedSample, significance_level: float
) -> tuple[float | None, float | None, float | None, str, list[str]]:
    """Student's paired t-test on the per-instance differences.

    Returns (statistic, p_value, degrees_of_freedom, test_used, warnings).
    """

    warnings: list[str] = []
    n = sample.n
    if n < 2:
        return None, None, None, "not_applicable_n_too_small", warnings
    mean = sum(sample.differences) / n
    variance = sum((value - mean) ** 2 for value in sample.differences) / (n - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        # Not a statistical question: every instance moved by the same amount.
        # In a benchmark that usually means the two methods differ by a constant
        # rather than by what they search, which is a bug worth looking at.
        warnings.append(
            f"Every one of the {n} paired differences is exactly {mean:.6g}. There is no variation "
            "to test, and two genuinely different search methods rarely differ by a constant on "
            "every instance -- check that the experiment is not computing one method from the other."
        )
        return None, None, None, "not_applicable_zero_variance", warnings

    standard_error = sd / math.sqrt(n)
    statistic = mean / standard_error
    degrees_of_freedom = float(n - 1)
    if _scipy_stats is not None:
        try:
            p_value = float(2.0 * _scipy_stats.t.sf(abs(statistic), degrees_of_freedom))
            return statistic, p_value, degrees_of_freedom, "paired_t", warnings
        except Exception:  # pragma: no cover - defensive
            pass
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, p_value, degrees_of_freedom, "paired_normal_approximation", warnings


def _wilcoxon_check(
    sample: PairedSample, t_significant: bool | None, significance_level: float
) -> tuple[float | None, list[str]]:
    """A distribution-free second opinion that may only ever raise doubt.

    Objective differences across a benchmark are rarely normal, so a rank test
    is worth running. It is deliberately not allowed to change the verdict: a
    procedure that reports whichever of two tests agrees with the hypothesis is
    not two tests, it is one biased test.
    """

    if _scipy_stats is None or sample.n < 6:
        return None, []
    non_zero = [value for value in sample.differences if value != 0]
    if len(non_zero) < 6:
        return None, []
    try:
        result = _scipy_stats.wilcoxon(non_zero, alternative="two-sided")
        p_value = float(result.pvalue)
    except Exception:  # pragma: no cover - defensive
        return None, []

    warnings: list[str] = []
    rank_significant = p_value < significance_level
    if t_significant is True and not rank_significant:
        warnings.append(
            f"The paired t-test is significant but the distribution-free Wilcoxon signed-rank test "
            f"is not (p={p_value:.4g}). The result leans on the normality assumption; treat it as "
            "weaker than the t-test alone suggests."
        )
    elif t_significant is False and rank_significant:
        warnings.append(
            f"A Wilcoxon signed-rank test on the same pairs is significant (p={p_value:.4g}) while "
            "the t-test is not, which can happen when a few outliers inflate the variance. The "
            "reported verdict stays with the t-test; this is a reason to look at the per-instance "
            "differences, not to claim the improvement."
        )
    return p_value, warnings


def _paired_effect_size(sample: PairedSample) -> float | None:
    """Cohen's d_z: mean difference over the sd of the differences."""

    n = sample.n
    if n < 2:
        return None
    mean = sum(sample.differences) / n
    variance = sum((value - mean) ** 2 for value in sample.differences) / (n - 1)
    sd = math.sqrt(variance)
    return mean / sd if sd > 0 else None


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
    design = "not_applicable"
    paired: PairedSample | None = None
    wilcoxon_p: float | None = None
    effective_n = min(item.n for item in stats)

    if proposed.method == baseline.method:
        warnings.append(
            "Baseline and proposed method resolved to the same method; no comparison was made. "
            "Set `baseline_method` and `proposed_method` in the config."
        )
    else:
        effect = (
            proposed.mean - baseline.mean if maximize else baseline.mean - proposed.mean
        )
        # The design is settled before any test runs. Whichever test that
        # choice implies is the one reported -- there is no second p-value to
        # fall back on if the first is inconvenient.
        paired, pairing_notes = detect_paired_sample(
            df, metric, baseline.method, proposed.method, maximize
        )
        warnings.extend(pairing_notes)
        if paired is not None:
            design = "paired"
            effective_n = paired.n
            statistic, p_value, degrees_of_freedom, test_used, paired_warnings = _paired_test(
                paired, significance_level
            )
            warnings.extend(paired_warnings)
            standardized = _paired_effect_size(paired)
            # The per-instance mean difference is the paired estimate of the
            # effect, and it is not the difference of the group means whenever
            # the benchmark is unbalanced.
            effect = sum(paired.differences) / paired.n
        else:
            design = "unpaired"
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

    if paired is not None:
        wilcoxon_p, wilcoxon_warnings = _wilcoxon_check(paired, significant, significance_level)
        warnings.extend(wilcoxon_warnings)
        ties = sum(1 for value in paired.differences if value == 0)
        if ties and ties >= 0.5 * paired.n:
            # Common when both methods reach a proved optimum on easy instances.
            # The test then rests on the handful of instances that separated
            # them, and the headline n overstates how much evidence there is.
            warnings.append(
                f"{ties} of {paired.n} instances are exact ties, so the comparison actually rests "
                f"on {paired.n - ties} instance(s). The benchmark may be too easy to separate these "
                "methods; report the win/loss/tie counts alongside the p-value."
            )

    min_n = effective_n
    if min_n < MIN_N_FOR_POWER:
        unit = "paired instance" if design == "paired" else "observation"
        warnings.append(
            f"The comparison rests on {min_n} {unit}(s) (< {MIN_N_FOR_POWER}); the test is "
            "underpowered. Add instance seeds or replications."
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
        test_statistic=statistic,
        approximate_p_value=p_value,
        degrees_of_freedom=degrees_of_freedom,
        test_used=test_used,
        design=design,
        pairing_key=paired.key if paired else [],
        n_pairs=paired.n if paired else None,
        n_wins=sum(1 for value in paired.differences if value > 0) if paired else None,
        n_losses=sum(1 for value in paired.differences if value < 0) if paired else None,
        n_ties=sum(1 for value in paired.differences if value == 0) if paired else None,
        paired_mean_difference=(sum(paired.differences) / paired.n) if paired else None,
        paired_sd_difference=_sd(paired.differences) if paired else None,
        wilcoxon_p_value=wilcoxon_p,
        significance_level=significance_level,
        is_statistically_significant=significant,
        evidence_strength=_classify(len(stats), effect, significant, min_n),
        warnings=warnings,
    )
    return _write(report, output_dir)


def _sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _write(report: StatisticalEvidenceReport, output_dir: str | Path) -> StatisticalEvidenceReport:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "statistical_evidence.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (root / "statistical_evidence.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
