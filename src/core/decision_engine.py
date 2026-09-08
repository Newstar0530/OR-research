from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.experiment_contract import ContractValidationReport
from src.schemas import ExecutionResult


def _truthy(series: pd.Series) -> pd.Series:
    """Interpret a CSV column that may be bool, string or numeric as a boolean mask."""

    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "t"})
    )


class DecisionEngine:
    """Turns one experiment's `results.csv` into a node status and a node metric.

    Two rules matter here and are enforced before any averaging happens:

    * A row whose solution failed independent verification is a **defect**, not
      a data point. Its presence fails the node outright, because averaging it
      away would hide a solver that reports solutions it cannot produce.
    * When `metric_method` is set, the node metric measures *that* method. A
      node metric taken over "whichever method did best" would be dominated by
      the exact reference solver and would rank every node identically.
    """

    def __init__(
        self,
        primary_metric: str = "objective",
        objective_direction: str = "minimize",
        metric_method: str | None = None,
        baseline_method: str | None = None,
        proposed_method: str | None = None,
    ) -> None:
        self.primary_metric = primary_metric
        self.maximize = objective_direction == "maximize"
        self.metric_method = metric_method
        self.baseline_method = baseline_method
        self.proposed_method = proposed_method

    # ------------------------------------------------------------------ helpers

    def _usable_rows(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
        """Drop unusable rows, or report a hard failure. Returns (rows, failure)."""

        if "verification_ok" in df.columns:
            broken = df[~_truthy(df["verification_ok"])]
            if not broken.empty:
                methods = sorted({str(m) for m in broken.get("method", [])})
                return df, (
                    f"{len(broken)} row(s) failed independent solution verification "
                    f"(methods: {', '.join(methods) or 'unknown'}). A reported solution that does "
                    "not satisfy the constraints is a defect, so this node is not usable."
                )
        rows = df
        if "feasible" in rows.columns:
            rows = rows[_truthy(rows["feasible"])]
        if "solver_status" in rows.columns:
            rows = rows[~rows["solver_status"].astype(str).isin({"not_run", "error"})]
        return rows, None

    # ------------------------------------------------------------------ metric

    def evaluate(
        self,
        results_csv: str | Path,
        execution: ExecutionResult,
        contract: ContractValidationReport,
    ) -> tuple[str, float | None, str]:
        if execution.status != "success":
            return execution.status, None, f"Execution ended with status {execution.status}."
        if not contract.passed:
            return "contract_failed", None, "Experiment output did not satisfy the generic experiment contract."
        path = Path(results_csv)
        if not path.exists():
            return "contract_failed", None, "results.csv was not produced."
        df = pd.read_csv(path)
        if self.primary_metric not in df.columns:
            return "contract_failed", None, f"Primary metric column `{self.primary_metric}` was not found."

        rows, failure = self._usable_rows(df)
        if failure:
            return "contract_failed", None, failure
        if rows.empty:
            return (
                "contract_failed",
                None,
                "No verified-feasible rows remained in results.csv after dropping unusable rows.",
            )

        scope_note = ""
        if self.metric_method and "method" in rows.columns:
            scoped = rows[rows["method"].astype(str) == self.metric_method]
            if scoped.empty:
                return (
                    "contract_failed",
                    None,
                    f"Primary metric is defined on method `{self.metric_method}`, which produced no "
                    "verified-feasible rows.",
                )
            rows = scoped
            scope_note = f" (restricted to method `{self.metric_method}`)"

        metric_series = pd.to_numeric(rows[self.primary_metric], errors="coerce").dropna()
        if metric_series.empty:
            return (
                "contract_failed",
                None,
                f"Primary metric `{self.primary_metric}` had no numeric values{scope_note}.",
            )

        if not self.metric_method and "method" in rows.columns:
            grouped = (
                rows.assign(_metric=pd.to_numeric(rows[self.primary_metric], errors="coerce"))
                .groupby("method")["_metric"]
                .mean()
                .dropna()
            )
            if not grouped.empty:
                best_method = grouped.idxmax() if self.maximize else grouped.idxmin()
                best_value = float(grouped.loc[best_method])
                return (
                    "success",
                    best_value,
                    f"Best mean {self.primary_metric} was achieved by method `{best_method}`.",
                )

        value = float(metric_series.mean())
        return (
            "success",
            value,
            f"Mean {self.primary_metric} was {value:.6g} over {len(metric_series)} "
            f"verified row(s){scope_note}.",
        )

    # -------------------------------------------------------------- comparison

    def _resolve_pair(self, methods: list[str]) -> tuple[str | None, str | None]:
        """Find the baseline and proposed method names, config first, names second."""

        baseline = self.baseline_method if self.baseline_method in methods else None
        proposed = self.proposed_method if self.proposed_method in methods else None
        if baseline is None:
            baseline = next((m for m in methods if "base" in m.lower() or "greedy" in m.lower()), None)
        if proposed is None:
            proposed = next(
                (
                    m
                    for m in methods
                    if m != baseline
                    and any(token in m.lower() for token in ("proposed", "adaptive", "local_search", "search"))
                ),
                None,
            )
        return baseline, proposed

    def method_comparison(self, results_csv: str | Path) -> dict[str, float | str | bool] | None:
        path = Path(results_csv)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "method" not in df.columns or self.primary_metric not in df.columns:
            return None
        rows, failure = self._usable_rows(df)
        if failure:
            return {
                "supports_hypothesis": False,
                "reason": failure,
                "verification_failed": True,
            }
        if rows.empty:
            return None
        grouped = (
            rows.assign(_metric=pd.to_numeric(rows[self.primary_metric], errors="coerce"))
            .groupby("method")["_metric"]
            .mean()
            .dropna()
        )
        if grouped.empty:
            return None
        methods = [str(method) for method in grouped.index]
        baseline, proposed = self._resolve_pair(methods)
        if baseline is None or proposed is None or baseline == proposed:
            best_method = grouped.idxmax() if self.maximize else grouped.idxmin()
            return {
                "best_method": str(best_method),
                "best_value": float(grouped.loc[best_method]),
                "supports_hypothesis": False,
                "reason": (
                    "Could not identify a distinct baseline and proposed method. Set "
                    "`baseline_method` and `proposed_method` in the config to make the comparison explicit."
                ),
            }
        baseline_value = float(grouped.loc[baseline])
        proposed_value = float(grouped.loc[proposed])
        improvement = (
            proposed_value - baseline_value if self.maximize else baseline_value - proposed_value
        )
        return {
            "baseline_method": baseline,
            "proposed_method": proposed,
            "baseline_value": baseline_value,
            "proposed_value": proposed_value,
            "improvement": improvement,
            "supports_hypothesis": improvement > 0,
            "reason": (
                f"Comparison on `{self.primary_metric}` over verified-feasible rows only; positive "
                "improvement means the proposed method outperformed the baseline under the configured "
                "objective direction."
            ),
        }
