"""
Temporal feature transformers.
Extracts calendar components and applies cyclical sin/cos encoding
to preserve the circular nature of periodic features.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class CalendarFeatures(BaseFeatureTransformer):
    """Extracts year, month, day, hour, weekday, and is_weekend from datetime columns."""

    def __init__(self, columns: List[str]):
        super().__init__(name="calendar_features")
        self.columns = columns
        self.valid_cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "CalendarFeatures":
        self.valid_cols = [c for c in self.columns if c in df.columns]
        self._output_columns = [
            f"{c}_{part}"
            for c in self.valid_cols
            for part in ["year", "month", "day", "hour", "weekday", "is_weekend"]
        ]
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col in self.valid_cols:
            dt = pd.to_datetime(df[col], errors="coerce")
            result[f"{col}_year"] = dt.dt.year.astype("Int64")
            result[f"{col}_month"] = dt.dt.month.astype("Int64")
            result[f"{col}_day"] = dt.dt.day.astype("Int64")
            result[f"{col}_hour"] = dt.dt.hour.astype("Int64")
            result[f"{col}_weekday"] = dt.dt.weekday.astype("Int64")
            result[f"{col}_is_weekend"] = dt.dt.weekday.isin([5, 6]).astype(int)
        return result


class CyclicalEncoder(BaseFeatureTransformer):
    """
    Encodes periodic features as (sin, cos) pairs.
    This preserves adjacency at period boundaries:
    e.g., December (month=12) is adjacent to January (month=1).
    Requires column names that end in a known period suffix
    (month, hour, weekday, day, minute).
    """

    PERIODS: Dict[str, int] = {
        "month": 12,
        "hour": 24,
        "weekday": 7,
        "day": 31,
        "minute": 60,
    }

    def __init__(self, columns: List[str]):
        super().__init__(name="cyclical_encoder")
        self.columns = columns
        self._col_periods: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "CyclicalEncoder":
        for col in self.columns:
            if col not in df.columns:
                continue
            # Infer period from column name suffix
            suffix = col.split("_")[-1]
            period = self.PERIODS.get(suffix)
            if period is None:
                logger.warning(f"CyclicalEncoder: unknown period for '{col}' — skipping")
                continue
            self._col_periods[col] = period
        self._output_columns = [
            f"{c}_{s}" for c in self._col_periods for s in ["sin", "cos"]
        ]
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col, period in self._col_periods.items():
            val = pd.to_numeric(df[col], errors="coerce").fillna(0)
            result[f"{col}_sin"] = np.sin(2 * np.pi * val / period)
            result[f"{col}_cos"] = np.cos(2 * np.pi * val / period)
        return result
