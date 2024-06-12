"""
Feature transformer registry.
Maps transformer type strings to their factory classes.
Used by FeaturePipeline to instantiate transformers from YAML config.
"""
from typing import Callable, Dict

from src.features.base import BaseFeatureTransformer
from src.features.numerical import (
    BinEncoder,
    InteractionFeatures,
    LogTransformer,
    NumericalScaler,
)
from src.features.categorical import FrequencyEncoder, OneHotEncoder, TargetEncoder
from src.features.temporal import CalendarFeatures, CyclicalEncoder

TRANSFORMER_REGISTRY: Dict[str, Callable[..., BaseFeatureTransformer]] = {
    "numerical_scaler": NumericalScaler,
    "log_transformer": LogTransformer,
    "bin_encoder": BinEncoder,
    "interaction_features": InteractionFeatures,
    "frequency_encoder": FrequencyEncoder,
    "target_encoder": TargetEncoder,
    "one_hot_encoder": OneHotEncoder,
    "calendar_features": CalendarFeatures,
    "cyclical_encoder": CyclicalEncoder,
}


def get_transformer(transformer_type: str, **kwargs) -> BaseFeatureTransformer:
    """Instantiate a transformer by its registry key with given kwargs."""
    if transformer_type not in TRANSFORMER_REGISTRY:
        raise ValueError(
            f"Unknown transformer: '{transformer_type}'. "
            f"Available: {sorted(TRANSFORMER_REGISTRY.keys())}"
        )
    return TRANSFORMER_REGISTRY[transformer_type](**kwargs)
