"""
Base class for all feature transformers.
Enforces a consistent fit/transform interface with metadata reporting.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseFeatureTransformer(ABC):
    """Abstract base defining the interface for all feature engineering components."""

    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False
        self._output_columns: List[str] = []

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseFeatureTransformer":
        """Fit transformer to the data. Must set _is_fitted = True."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted state. Must not modify fit state."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit then transform."""
        return self.fit(df).transform(df)

    @property
    def output_columns(self) -> List[str]:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before accessing output_columns")
        return self._output_columns

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_fitted": self._is_fitted,
            "n_output_features": len(self._output_columns),
            "output_columns": self._output_columns,
        }
