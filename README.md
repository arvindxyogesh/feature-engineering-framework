# Feature Engineering Framework

A **reusable, config-driven feature engineering and data validation framework** for tabular ML. Supports numerical, categorical, and temporal transformers under a consistent interface, with full pipeline persistence and a comprehensive test suite.

## Architecture

```
feature-engineering-framework/
├── config/
│   └── feature_config.yaml         # Declare transformers in YAML — no code changes
├── data/
│   └── generate_data.py            # Synthetic e-commerce dataset generator
├── src/
│   ├── features/
│   │   ├── base.py                 # BaseFeatureTransformer (fit/transform interface)
│   │   ├── numerical.py            # NumericalScaler, LogTransformer, BinEncoder, Interactions
│   │   ├── categorical.py          # FrequencyEncoder, TargetEncoder, OneHotEncoder
│   │   ├── temporal.py             # CalendarFeatures, CyclicalEncoder
│   │   └── registry.py             # String → class mapping for YAML instantiation
│   ├── validation/
│   │   └── validator.py            # Schema, missingness, cardinality, row count checks
│   └── pipeline/
│       └── feature_pipeline.py     # Validate → fit → transform → persist
├── artifacts/                      # CSV feature matrices + pickled pipeline
├── scripts/
│   └── run_feature_engineering.py  # CLI entry point
└── tests/
    └── test_features.py            # Unit tests for every transformer
```

## Available Transformers

| Type key | Class | Description |
|---|---|---|
| `numerical_scaler` | `NumericalScaler` | StandardScaler / MinMaxScaler / RobustScaler |
| `log_transformer` | `LogTransformer` | log1p for right-skewed non-negative features |
| `bin_encoder` | `BinEncoder` | Quantile-based integer binning |
| `interaction_features` | `InteractionFeatures` | Pairwise multiplicative interactions |
| `frequency_encoder` | `FrequencyEncoder` | Category frequency in training set |
| `target_encoder` | `TargetEncoder` | Smoothed mean-target encoding |
| `one_hot_encoder` | `OneHotEncoder` | OHE with max cardinality guard |
| `calendar_features` | `CalendarFeatures` | year, month, day, hour, weekday, is_weekend |
| `cyclical_encoder` | `CyclicalEncoder` | sin/cos for month, hour, weekday |

## Quick Start

```bash
pip install -r requirements.txt

# Generate data and run the full pipeline
python scripts/run_feature_engineering.py
```

## Config-Driven Design

Add, remove, or reorder transformers entirely in YAML:

```yaml
transformers:
  - type: log_transformer
    columns: [cart_value, session_duration_sec]

  - type: target_encoder
    columns: [device_type, country]
    smoothing: 15.0

  - type: cyclical_encoder
    columns: [session_start_hour, session_start_weekday]
```

## Key Design Principles

| Principle | Implementation |
|---|---|
| **No data leakage** | `fit_transform` on training split only; `transform` on test |
| **Consistent interface** | Every transformer: `fit` → `transform` → `output_columns` |
| **Validation first** | Schema + missingness + cardinality checked before any transform |
| **Pipeline persistence** | `pipeline.save(path)` / `FeaturePipeline.load(path)` |
| **Config-driven** | Swap transformers via YAML without touching Python |

## Running Tests

```bash
pytest tests/ -v
```

## Output Artifacts

| File | Description |
|---|---|
| `artifacts/train_features.csv` | Engineered training feature matrix |
| `artifacts/test_features.csv` | Engineered test feature matrix |
| `artifacts/feature_pipeline.pkl` | Serialized fitted pipeline for inference |
