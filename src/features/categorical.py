"""
Categorical feature transformers.
Includes frequency encoding, smoothed target encoding, and one-hot encoding.
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class FrequencyEncoder(BaseFeatureTransformer):
    """Encodes categories by their relative frequency in the training dataset."""

    def __init__(self, columns: List[str]):
        super().__init__(name="frequency_encoder")
        self.columns = columns
        self._freq_maps: Dict[str, Dict] = {}

    def fit(self, df: pd.DataFrame) -> "FrequencyEncoder":
        for col in self.columns:
            if col in df.columns:
                self._freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        self._output_columns = [f"{c}_freq" for c in self._freq_maps]
        self._is_fitted = True
        logger.info(f"FrequencyEncoder: {len(self._freq_maps)} columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col, freq_map in self._freq_maps.items():
            result[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)
        return result


class TargetEncoder(BaseFeatureTransformer):
    """
    Smoothed mean target encoding.
    Uses additive smoothing: (count * mean + smoothing * global_mean) / (count + smoothing).
    This regularizes rare categories toward the global mean, preventing overfitting.
    """

    def __init__(self, columns: List[str], target_col: str, smoothing: float = 10.0):
        super().__init__(name="target_encoder")
        self.columns = columns
        self.target_col = target_col
        self.smoothing = smoothing
        self._encoding_maps: Dict[str, Dict] = {}
        self._global_mean: float = 0.0

    def fit(self, df: pd.DataFrame) -> "TargetEncoder":
        self._global_mean = float(df[self.target_col].mean())
        for col in self.columns:
            if col not in df.columns:
                continue
            stats = df.groupby(col)[self.target_col].agg(["mean", "count"])
            smoothed = (
                (stats["count"] * stats["mean"] + self.smoothing * self._global_mean)
                / (stats["count"] + self.smoothing)
            )
            self._encoding_maps[col] = smoothed.to_dict()
        self._output_columns = [f"{c}_target_enc" for c in self._encoding_maps]
        self._is_fitted = True
        logger.info(
            f"TargetEncoder: {len(self._encoding_maps)} columns "
            f"(global_mean={self._global_mean:.4f}, smoothing={self.smoothing})"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col, enc_map in self._encoding_maps.items():
            result[f"{col}_target_enc"] = df[col].map(enc_map).fillna(self._global_mean)
        return result


class OneHotEncoder(BaseFeatureTransformer):
    """One-hot encodes categorical columns, capped at max_cardinality categories."""

    def __init__(self, columns: List[str], max_cardinality: int = 20):
        super().__init__(name="one_hot_encoder")
        self.columns = columns
        self.max_cardinality = max_cardinality
        self._categories: Dict[str, List] = {}

    def fit(self, df: pd.DataFrame) -> "OneHotEncoder":
        for col in self.columns:
            if col in df.columns:
                cats = df[col].value_counts().head(self.max_cardinality).index.tolist()
                self._categories[col] = cats
        self._output_columns = [
            f"{col}_ohe_{str(cat).replace(' ', '_')}"
            for col, cats in self._categories.items()
            for cat in cats
        ]
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col, cats in self._categories.items():
            for cat in cats:
                safe_name = str(cat).replace(" ", "_")
                result[f"{col}_ohe_{safe_name}"] = (df[col] == cat).astype(int)
        return result
