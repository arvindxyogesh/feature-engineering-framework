"""
Feature engineering pipeline entry point.
Generates synthetic data, runs the full pipeline, and saves artifacts.

Usage:
    cd feature-engineering-framework
    python scripts/run_feature_engineering.py
"""
import logging
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    from data.generate_data import generate_ecommerce_data
    from src.pipeline.feature_pipeline import FeaturePipeline

    with open("config/feature_config.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("Generating synthetic e-commerce dataset...")
    df = generate_ecommerce_data(n_samples=5000)
    logger.info(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Purchase rate: {df['purchased'].mean():.2%}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    pipeline = FeaturePipeline(config)

    logger.info("Fitting pipeline on training data...")
    train_features, _ = pipeline.fit_transform(train_df.copy())

    logger.info("Transforming test data...")
    test_features, _ = pipeline.transform(test_df.copy())

    feature_names = pipeline.get_feature_names()

    print("\n" + "=" * 55)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 55)
    print(f"  Input columns:    {len(df.columns)}")
    print(f"  Output features:  {len(feature_names)}")
    print(f"  Train shape:      {train_features.shape}")
    print(f"  Test shape:       {test_features.shape}")
    print(f"\nFeature sample (first 20):")
    for name in feature_names[:20]:
        print(f"  {name}")
    if len(feature_names) > 20:
        print(f"  ... and {len(feature_names) - 20} more")

    os.makedirs("artifacts", exist_ok=True)
    train_features.to_csv("artifacts/train_features.csv", index=False)
    test_features.to_csv("artifacts/test_features.csv", index=False)
    pipeline.save("artifacts/feature_pipeline.pkl")

    print(f"\nArtifacts:")
    print(f"  artifacts/train_features.csv")
    print(f"  artifacts/test_features.csv")
    print(f"  artifacts/feature_pipeline.pkl")


if __name__ == "__main__":
    main()
