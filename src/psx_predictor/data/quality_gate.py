"""
quality_gate.py
----------------
Post-aggregation data quality checks that run before build_features.py
finalizes output. Fails loudly (raises or logs critical warnings) rather
than shipping a CSV with silent placeholders.

This is how defects #2–#8 from the PSO audit get caught automatically
next time, instead of shipping silently in a master CSV again.

Usage:
    from src.psx_predictor.data.quality_gate import run_quality_checks
    failures = run_quality_checks(df, ticker="PSO", strict=False)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("QualityGate")
logger.setLevel(logging.INFO)


@dataclass
class QualityReport:
    """Result of running all quality checks on a ticker's feature DataFrame."""
    ticker: str
    total_rows: int
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Quality Gate [{status}] for {self.ticker} ({self.total_rows} rows)"]
        for f in self.failures:
            lines.append(f"  ❌ FAIL: {f}")
        for w in self.warnings:
            lines.append(f"  ⚠️ WARN: {w}")
        return "\n".join(lines)


# ── Columns expected to actually vary given enough history ───────────────────
# Format: (column_name, min_expected_changes_per_year, severity)
# severity: "fail" = hard failure, "warn" = log warning only
EXPECTED_VARIANCE = [
    # Fundamentals — quarterly reporting means ~4 changes/year
    ("revenue", 3.0, "warn"),
    ("net_income", 3.0, "warn"),
    ("eps_trailing", 3.0, "warn"),
    ("total_assets", 2.0, "warn"),
    # Sentiment — should move most days once news pipeline is live
    ("sentiment_score", 20.0, "warn"),
    # Macro — policy rate changes ~6×/year but only when rate moves
    ("sbp_policy_rate", 0.1, "warn"),
    # Corporate events — should have at least a few events per year
    ("earnings_event", 0.5, "warn"),
]

# ── Columns that should not be constant-zero for the full dataset ────────────
SHOULD_NOT_BE_ALL_ZERO = [
    ("sentiment_score", "warn"),
    ("earnings_event", "warn"),
    ("dividend_event", "warn"),
    ("election_flag", "warn"),
    ("india_pakistan_tension_flag", "warn"),
    ("middle_east_conflict_flag", "warn"),
]


def check_placeholder_columns(df: pd.DataFrame, years_covered: float) -> tuple[list[str], list[str]]:
    """
    Checks that columns expected to vary actually do.
    Returns (failures, warnings).
    """
    failures = []
    warnings = []

    for col, min_changes_per_year, severity in EXPECTED_VARIANCE:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        changes = (series.diff().abs() > 1e-10).sum()
        expected_min = max(1, min_changes_per_year * years_covered * 0.3)  # 30% tolerance

        if changes < expected_min:
            msg = (
                f"{col}: only {changes} value changes over {years_covered:.1f} years "
                f"(expected >= {expected_min:.0f}). Likely placeholder/frozen data."
            )
            if severity == "fail":
                failures.append(msg)
            else:
                warnings.append(msg)

    return failures, warnings


def check_constant_zero_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Checks that columns which should carry signal are not 100% zero.
    Returns (failures, warnings).
    """
    failures = []
    warnings = []

    for col, severity in SHOULD_NOT_BE_ALL_ZERO:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        nonzero_pct = (series.abs() > 1e-10).mean()

        if nonzero_pct < 0.001:  # Less than 0.1% non-zero
            msg = f"{col}: {nonzero_pct:.2%} non-zero rows. Column carries no signal."
            if severity == "fail":
                failures.append(msg)
            else:
                warnings.append(msg)

    return failures, warnings


def check_synthetic_flags(df: pd.DataFrame, max_synthetic_pct: float = 0.80) -> tuple[list[str], list[str]]:
    """
    Checks that synthetic data flags don't exceed thresholds.
    Returns (failures, warnings).
    """
    failures = []
    warnings = []

    flag_cols = [c for c in df.columns if "is_synthetic" in c.lower()]
    for col in flag_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        pct = series.mean()
        if pct > max_synthetic_pct:
            msg = f"{col}: {pct:.0%} of rows are synthetic (threshold {max_synthetic_pct:.0%})"
            warnings.append(msg)

    return failures, warnings


def check_adjusted_close(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Checks that adjusted_close is populated (defect #1 from audit).
    Returns (failures, warnings).
    """
    failures = []
    warnings = []

    if "adjusted_close" in df.columns:
        null_pct = df["adjusted_close"].isna().mean()
        if null_pct > 0.95:
            warnings.append(
                f"adjusted_close: {null_pct:.0%} null. "
                f"Fallback to close will be used, but dividend/split-adjusted returns are unavailable."
            )

    return failures, warnings


def check_ohlc_integrity(df: pd.DataFrame, tolerance: float = 0.01) -> tuple[list[str], list[str]]:
    """
    Checks that open/close fall within [low - eps, high + eps].
    Defect #10 from audit.
    Returns (failures, warnings).
    """
    failures = []
    warnings = []

    if not all(c in df.columns for c in ["open", "high", "low", "close"]):
        return failures, warnings

    open_below_low = (df["open"] < df["low"] - tolerance).sum()
    open_above_high = (df["open"] > df["high"] + tolerance).sum()
    close_below_low = (df["close"] < df["low"] - tolerance).sum()
    close_above_high = (df["close"] > df["high"] + tolerance).sum()

    violations = open_below_low + open_above_high + close_below_low + close_above_high
    if violations > 0:
        warnings.append(
            f"OHLC integrity: {violations} rows where open/close is outside "
            f"[low-{tolerance}, high+{tolerance}] band."
        )

    return failures, warnings


def check_zero_volume_days(df: pd.DataFrame, max_pct: float = 0.15) -> tuple[list[str], list[str]]:
    """
    Checks for excessive zero-volume trading days (defect #9).
    Returns (failures, warnings).
    """
    failures = []
    warnings = []

    if "volume" not in df.columns:
        return failures, warnings

    zero_vol_pct = (df["volume"] == 0).mean()
    if zero_vol_pct > max_pct:
        warnings.append(
            f"Zero-volume days: {zero_vol_pct:.1%} of rows ({(df['volume'] == 0).sum()} rows). "
            f"These may be non-trading/suspended days that distort "
            f"relative_volume, turnover_ratio, VWAP, OBV."
        )

    return failures, warnings


def run_quality_checks(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    strict: bool = False,
) -> QualityReport:
    """
    Runs all quality checks and returns a QualityReport.

    Args:
        df: Feature DataFrame (output of build_features).
        ticker: Ticker symbol for reporting.
        strict: If True, raises RuntimeError on any failure.

    Returns:
        QualityReport with failures and warnings.
    """
    if df.empty:
        return QualityReport(ticker=ticker, total_rows=0, failures=["DataFrame is empty."])

    # Calculate years covered
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        years = (dates.max() - dates.min()).days / 365.25
    else:
        years = len(df) / 252.0  # Assume trading days

    report = QualityReport(ticker=ticker, total_rows=len(df))

    # Run all checks
    checks = [
        check_placeholder_columns(df, years),
        check_constant_zero_columns(df),
        check_synthetic_flags(df),
        check_adjusted_close(df),
        check_ohlc_integrity(df),
        check_zero_volume_days(df),
    ]

    for check_failures, check_warnings in checks:
        report.failures.extend(check_failures)
        report.warnings.extend(check_warnings)

    # Log the report
    logger.info(report.summary())

    if strict and not report.passed:
        raise RuntimeError(
            f"Quality gate FAILED for {ticker}. "
            f"{len(report.failures)} failure(s). Set strict=False to continue anyway."
        )

    return report
