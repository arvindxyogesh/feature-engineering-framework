"""
Numerical feature transformers.
Includes StandardScaler/MinMax/Robust scaling, log transform,
quantile binning, and pairwise interaction terms.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.features.base import BaseFeatureTransformer

logger = logging.getLogger(__name__)

SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class NumericalScaler(BaseFeatureTransformer):
    """Scales numerical columns using a configurable sklearn scaler."""

    def __init__(self, columns: List[str], method: str = "standard"):
        super().__init__(name=f"numerical_scaler_{method}")
        self.columns = columns
        self.method = method
        if method not in SCALER_MAP:
            raise ValueError(f"Unknown scale method: {method}. Choose from {list(SCALER_MAP)}")
        self._scalers: Dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "NumericalScaler":
        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not in dataframe — skipping")
                continue
            scaler = SCALER_MAP[self.method]()
            scaler.fit(df[[col]].dropna())
            self._scalers[col] = scaler
        self._output_columns = [f"{c}_{self.method}_scaled" for c in self._scalers]
        self._is_fitted = True
        logger.info(f"{self.name}: fitted on {len(self._scalers)} columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col, scaler in self._scalers.items():
            vals = df[[col]].copy()
            vals[col] = vals[col].fillna(vals[col].median())
            result[f"{col}_{self.method}_scaled"] = scaler.transform(vals).flatten()
        return result


class LogTransformer(BaseFeatureTransformer):
    """Applies log1p to right-skewed non-negative columns."""

    def __init__(self, columns: List[str]):
        super().__init__(name="log_transformer")
        self.columns = columns
        self.valid_cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "LogTransformer":
        self.valid_cols = [
            c for c in self.columns
            if c in df.columns and (df[c].dropna() >= 0).all()
        ]
        skipped = set(self.columns) - set(self.valid_cols)
        if skipped:
            logger.warning(f"LogTransformer: skipped {skipped} (not found or contain negatives)")
        self._output_columns = [f"{c}_log1p" for c in self.valid_cols]
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col in self.valid_cols:
            result[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0).fillna(0))
        return result


class BinEncoder(BaseFeatureTransformer):
    """Discretizes continuous features into quantile-based integer bins."""

    def __init__(self, columns: List[str], n_bins: int = 5):
        super().__init__(name="bin_encoder")
        self.columns = columns
        self.n_bins = n_bins
        self._bin_edges: Dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> "BinEncoder":
        for col in self.columns:
            if col in df.columns:
                _, edges = pd.qcut(
                    df[col].dropna(), q=self.n_bins,
                    retbins=True, duplicates="drop",
                )
                self._bin_edges[col] = edges
        self._output_columns = [f"{c}_bin" for c in self._bin_edges]
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col, edges in self._bin_edges.items():
            result[f"{col}_bin"] = pd.cut(
                df[col], bins=edges, labels=False, include_lowest=True
            ).astype("Int64")
        return result


class InteractionFeatures(BaseFeatureTransformer):
    """Creates pairwise multiplicative interaction terms between numerical columns."""

    def __init__(self, columns: List[str]):
        super().__init__(name="interaction_features")
        self.columns = columns
        self._pairs: List[tuple] = []

    def fit(self, df: pd.DataFrame) -> "InteractionFeatures":
        valid = [c for c in self.columns if c in df.columns]
        self._pairs = [
            (valid[i], valid[j])
            for i in range(len(valid))
            for j in range(i + 1, len(valid))
        ]
        self._output_columns = [f"{a}_x_{b}" for a, b in self._pairs]
        self._is_fitted = True
        logger.info(f"InteractionFeatures: {len(self._pairs)} interaction pairs")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for a, b in self._pairs:
            result[f"{a}_x_{b}"] = df[a].fillna(0) * df[b].fillna(0)
        return result
