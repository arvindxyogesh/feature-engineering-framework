"""
Data validation module.
Checks schema, data types, missing values, duplicates, and cardinality
before any feature engineering begins.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for the outcome of a validation run."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def report(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Data Validation: {status}"]
        lines += [f"  [ERROR]  {e}" for e in self.errors]
        lines += [f"  [WARN]   {w}" for w in self.warnings]
        if self.stats:
            lines.append(
                f"  Stats: {self.stats['n_rows']} rows, "
                f"{self.stats['n_columns']} cols, "
                f"{self.stats['n_missing_total']} missing values"
            )
        return "\n".join(lines)


class DataValidator:
    """Validates a DataFrame against schema and statistical expectations."""

    def __init__(self, config: dict):
        self.config = config.get("validation", {})

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Run all checks and return a ValidationResult."""
        errors, warnings = [], []

        # 1. Required columns
        required = self.config.get("required_columns", [])
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # 2. Minimum row count
        min_rows = self.config.get("min_rows", 10)
        if len(df) < min_rows:
            errors.append(f"Only {len(df)} rows (minimum: {min_rows})")

        # 3. Missing value thresholds
        max_miss = self.config.get("max_missing_pct", 0.3)
        for col in df.columns:
            miss_pct = df[col].isna().mean()
            if miss_pct > max_miss:
                errors.append(
                    f"'{col}': {miss_pct:.1%} missing (threshold: {max_miss:.1%})"
                )
            elif miss_pct > 0.05:
                warnings.append(f"'{col}': {miss_pct:.1%} missing values")

        # 4. Duplicate rows
        n_dupes = int(df.duplicated().sum())
        if n_dupes > 0:
            warnings.append(f"{n_dupes} duplicate rows detected")

        # 5. High-cardinality object columns (potential ID leakage)
        max_card = self.config.get("max_cardinality", 0.9)
        for col in df.select_dtypes(include="object").columns:
            card_ratio = df[col].nunique() / len(df)
            if card_ratio > max_card:
                warnings.append(
                    f"'{col}': very high cardinality ({card_ratio:.1%} unique values) — "
                    f"consider dropping or target-encoding"
                )

        stats = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_missing_total": int(df.isna().sum().sum()),
            "n_duplicates": n_dupes,
        }

        result = ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
        logger.info(result.report())
        return result
