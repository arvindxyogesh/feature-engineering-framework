"""
Feature pipeline orchestrator.
Chains a sequence of feature transformers configured via YAML.
Fits on training data only, then transforms any split consistently.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.features.registry import get_transformer
from src.features.base import BaseFeatureTransformer
from src.validation.validator import DataValidator

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrates the full feature engineering process:
    validate → fit all transformers → transform.
    """

    def __init__(self, config: dict):
        self.config = config
        self.validator = DataValidator(config)
        self.transformers: List[BaseFeatureTransformer] = []
        self._is_fitted = False
        self._build_from_config()

    def _build_from_config(self) -> None:
        """Instantiate all transformers from the YAML config list."""
        target_col = self.config.get("target_column")
        for t_cfg in self.config.get("transformers", []):
            kwargs = {k: v for k, v in t_cfg.items() if k != "type"}
            # TargetEncoder needs the target column name
            if t_cfg["type"] == "target_encoder":
                kwargs["target_col"] = target_col
            self.transformers.append(get_transformer(t_cfg["type"], **kwargs))
        logger.info(f"Pipeline: {len(self.transformers)} transformers configured")

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate data, fit all transformers on df, and return feature DataFrame.
        Returns: (feature_df, augmented_df) where augmented_df = original + features.
        """
        val_result = self.validator.validate(df)
        if not val_result.passed:
            raise ValueError(f"Validation failed:\n{val_result.report()}")

        feature_frames = []
        for transformer in self.transformers:
            try:
                features = transformer.fit_transform(df)
                feature_frames.append(features)
                logger.info(
                    f"  {transformer.name}: +{len(transformer.output_columns)} features"
                )
            except Exception as exc:
                logger.error(f"Transformer '{transformer.name}' failed: {exc}")
                raise

        self._is_fitted = True
        feature_df = pd.concat(feature_frames, axis=1)
        logger.info(f"Pipeline fit_transform: {feature_df.shape[1]} total features")
        return feature_df, pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform unseen data with all fitted transformers."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fit before transform")
        feature_frames = [t.transform(df) for t in self.transformers]
        feature_df = pd.concat(feature_frames, axis=1)
        return feature_df, pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    def get_feature_names(self) -> List[str]:
        """Return all output feature names across all transformers."""
        names = []
        for t in self.transformers:
            names.extend(t.output_columns)
        return names

    def save(self, path: str) -> None:
        """Serialize the fitted pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Load a previously saved pipeline from disk."""
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Pipeline loaded from {path}")
        return pipeline
