"""What a generated experiment must produce before its numbers count.

Two contracts live here:

* `LEGACY_CONTRACT` -- the original minimum (a `results.csv` with identity,
  method, objective, runtime and seed columns). Unchanged, so every existing
  template and test keeps passing.
* `SOLVER_CONTRACT` -- applied automatically when the results carry a
  `solver_status` column, i.e. when the experiment actually solved something.
  It is strictly harder: a claimed solution that failed independent feasibility
  verification is a contract **failure**, not a warning, and the promised
  `SUMMARY_JSON:` stdout line is finally checked instead of merely declared.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from src.core.solve_result import CLAIMS_SOLUTION


#: Checked in order; the experiment's own log is authoritative, but a run
#: directory may only have the generic subprocess log.
LOG_CANDIDATES = (
    "experiment_last.log",
    "subprocess_last.log",
    "experiment_stdout.log",
)

_TRUTHY = {"true", "1", "1.0", "yes", "t"}


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(_TRUTHY)


class ExperimentContract(BaseModel):
    """Minimum output contract for a generated OR experiment."""

    name: str = "legacy"
    required_columns: list[str] = Field(
        default_factory=lambda: ["instance_id", "method", "objective", "runtime_seconds", "seed"]
    )
    optional_columns: list[str] = Field(
        default_factory=lambda: [
            "problem_size",
            "feasible",
            "baseline_objective",
            "gap",
            "status",
            "replicate",
            "constraint_violation",
            "robustness_metric",
        ]
    )
    required_files: list[str] = Field(default_factory=lambda: ["results.csv"])
    required_stdout_prefix: str = "SUMMARY_JSON:"
    #: When True, a missing/unverifiable stdout line fails instead of warning.
    enforce_stdout: bool = False
    #: When True, verification and feasibility columns are checked as failures.
    enforce_verification: bool = False


LEGACY_CONTRACT = ExperimentContract()

SOLVER_CONTRACT = ExperimentContract(
    name="solver",
    required_columns=[
        "instance_id",
        "method",
        "objective",
        "runtime_seconds",
        "seed",
        "objective_sense",
        "solver_status",
        "feasible",
        "verification_ok",
        "n_vars",
        "n_constraints",
    ],
    optional_columns=[
        "problem_size",
        "instance_family",
        "solver_backend",
        "dual_bound",
        "mip_gap",
        "node_count",
        "hit_time_limit",
        "max_violation",
        "constraint_violation",
        "n_violated_constraints",
        "known_optimum",
        "known_optimum_source",
        "gap_to_known_optimum",
        "gap",
        "notes",
    ],
    enforce_stdout=True,
    enforce_verification=True,
)


class ContractValidationReport(BaseModel):
    passed: bool
    contract_name: str = "legacy"
    missing_files: list[str] = Field(default_factory=list)
    missing_columns: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        status = "passed" if self.passed else "failed"
        lines = [
            "# Experiment Contract Validation",
            "",
            f"Status: {status}",
            "",
            f"Contract: {self.contract_name}",
            "",
        ]
        if self.missing_files:
            lines.append("Missing files:")
            lines.extend(f"- {item}" for item in self.missing_files)
            lines.append("")
        if self.missing_columns:
            lines.append("Missing columns:")
            lines.extend(f"- {item}" for item in self.missing_columns)
            lines.append("")
        if self.failures:
            lines.append("Failures:")
            lines.extend(f"- {item}" for item in self.failures)
            lines.append("")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
            lines.append("")
        return "\n".join(lines)


def detect_contract(columns: list[str]) -> ExperimentContract:
    """Pick the strictest contract the results are able to satisfy."""

    return SOLVER_CONTRACT if "solver_status" in columns else LEGACY_CONTRACT


def _read_captured_stdout(root: Path) -> list[str]:
    """Every captured execution log we can find, so no single file can mask the rest."""

    captured: list[str] = []
    for name in LOG_CANDIDATES:
        candidate = root / name
        if not candidate.exists():
            continue
        try:
            captured.append(candidate.read_text(encoding="utf-8", errors="ignore"))
        except OSError:  # pragma: no cover - defensive
            continue
    return captured


def _check_stdout(
    root: Path, contract: ExperimentContract, failures: list[str], warnings: list[str]
) -> None:
    if not contract.required_stdout_prefix:
        return
    captured = _read_captured_stdout(root)
    if not captured:
        warnings.append(
            "Execution stdout was not captured next to the results, so the promised "
            f"`{contract.required_stdout_prefix}` line could not be verified."
        )
        return
    if any(contract.required_stdout_prefix in text for text in captured):
        return
    message = (
        f"Execution output does not contain the required `{contract.required_stdout_prefix}` "
        "summary line."
    )
    (failures if contract.enforce_stdout else warnings).append(message)


def _check_verification(
    df: pd.DataFrame, failures: list[str], warnings: list[str]
) -> None:
    """The core correctness gate for solver-backed experiments."""

    if "verification_ok" in df.columns:
        broken = df[~_truthy_mask(df["verification_ok"])]
        if not broken.empty:
            methods = sorted({str(m) for m in broken.get("method", [])})
            failures.append(
                f"{len(broken)} row(s) failed independent solution verification "
                f"(methods: {', '.join(methods) or 'unknown'}). A reported solution that does not "
                "satisfy the constraints invalidates the experiment."
            )

    if {"solver_status", "feasible"}.issubset(df.columns):
        claims = df[df["solver_status"].astype(str).isin(CLAIMS_SOLUTION)]
        infeasible = claims[~_truthy_mask(claims["feasible"])]
        if not infeasible.empty:
            failures.append(
                f"{len(infeasible)} row(s) report a solver status that claims a solution "
                "while the feasibility check says the solution is infeasible."
            )
        usable = claims[_truthy_mask(claims["feasible"])]
        if usable.empty:
            failures.append(
                "No row produced a verified-feasible solution, so there is nothing to analyze."
            )

    if "solver_status" in df.columns:
        statuses = df["solver_status"].astype(str)
        if not statuses.isin({"optimal"}).any():
            warnings.append(
                "No row proved optimality, so every reported gap is relative to an unproved "
                "reference. Treat quality comparisons as indicative only."
            )
        not_run = sorted({str(m) for m in df.loc[statuses == "not_run", "method"]}) if "method" in df.columns else []
        if not_run:
            warnings.append(
                f"These requested methods did not run on this machine: {', '.join(not_run)}. "
                "See the solver backend report for the reason."
            )
        errored = sorted({str(m) for m in df.loc[statuses == "error", "method"]}) if "method" in df.columns else []
        if errored:
            warnings.append(f"These methods ended in an error status: {', '.join(errored)}.")

    if "known_optimum" in df.columns and int(df["known_optimum"].notna().sum()) == 0:
        warnings.append(
            "No instance has a recorded proved optimum, so `gap_to_known_optimum` is unavailable."
        )


def validate_experiment_contract(
    run_dir: str | Path, contract: ExperimentContract | None = None
) -> ContractValidationReport:
    root = Path(run_dir)
    missing_files: list[str] = []
    missing_columns: list[str] = []
    failures: list[str] = []
    warnings: list[str] = []

    results_path = root / "results.csv"
    df: pd.DataFrame | None = None
    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
        except Exception as exc:
            warnings.append(f"Could not parse results.csv: {exc}")

    resolved = contract or detect_contract(list(df.columns) if df is not None else [])
    missing_files = [name for name in resolved.required_files if not (root / name).exists()]

    if df is not None:
        missing_columns = [col for col in resolved.required_columns if col not in df.columns]
        if df.empty:
            warnings.append("results.csv is empty.")
        if "method" in df.columns and df["method"].nunique() < 2:
            warnings.append(
                "Only one method appears in results.csv; baseline comparison may be weak."
            )
        if resolved.enforce_verification and not missing_columns:
            _check_verification(df, failures, warnings)

    _check_stdout(root, resolved, failures, warnings)

    return ContractValidationReport(
        passed=not missing_files and not missing_columns and not failures,
        contract_name=resolved.name,
        missing_files=missing_files,
        missing_columns=missing_columns,
        failures=failures,
        warnings=warnings,
    )
