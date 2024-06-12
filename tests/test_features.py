"""Tests for all feature transformers and the data validator."""
import numpy as np
import pandas as pd
import pytest

from src.features.numerical import NumericalScaler, LogTransformer, BinEncoder, InteractionFeatures
from src.features.categorical import FrequencyEncoder, TargetEncoder, OneHotEncoder
from src.features.temporal import CalendarFeatures, CyclicalEncoder
from src.validation.validator import DataValidator


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "age": rng.uniform(18, 65, n),
        "revenue": rng.exponential(100, n),
        "visits": rng.integers(1, 50, n).astype(float),
        "device": rng.choice(["mobile", "desktop", "tablet"], n),
        "source": rng.choice(["organic", "paid", "email"], n),
        "session_start": pd.date_range("2023-01-01", periods=n, freq="h"),
        "purchased": rng.integers(0, 2, n),
    })


# ── Numerical ───────────────────────────────────────────────────────────────

def test_numerical_scaler_standard(sample_df):
    scaler = NumericalScaler(columns=["age", "revenue"], method="standard")
    result = scaler.fit_transform(sample_df)
    assert "age_standard_scaled" in result.columns
    assert abs(result["age_standard_scaled"].mean()) < 0.05  # approx zero mean


def test_numerical_scaler_robust(sample_df):
    scaler = NumericalScaler(columns=["age"], method="robust")
    result = scaler.fit_transform(sample_df)
    assert "age_robust_scaled" in result.columns


def test_log_transformer(sample_df):
    lt = LogTransformer(columns=["revenue"])
    result = lt.fit_transform(sample_df)
    assert "revenue_log1p" in result.columns
    assert not result["revenue_log1p"].isna().any()


def test_bin_encoder(sample_df):
    be = BinEncoder(columns=["age"], n_bins=4)
    result = be.fit_transform(sample_df)
    assert "age_bin" in result.columns
    non_null = result["age_bin"].dropna()
    assert non_null.between(0, 3).all()


def test_interaction_features(sample_df):
    intf = InteractionFeatures(columns=["age", "revenue", "visits"])
    result = intf.fit_transform(sample_df)
    assert "age_x_revenue" in result.columns
    assert "age_x_visits" in result.columns
    assert "revenue_x_visits" in result.columns


# ── Categorical ──────────────────────────────────────────────────────────────

def test_frequency_encoder(sample_df):
    fe = FrequencyEncoder(columns=["device", "source"])
    result = fe.fit_transform(sample_df)
    assert "device_freq" in result.columns
    assert result["device_freq"].between(0.0, 1.0).all()


def test_target_encoder(sample_df):
    te = TargetEncoder(columns=["device"], target_col="purchased", smoothing=5.0)
    result = te.fit_transform(sample_df)
    assert "device_target_enc" in result.columns
    assert result["device_target_enc"].between(0.0, 1.0).all()


def test_one_hot_encoder(sample_df):
    ohe = OneHotEncoder(columns=["device"], max_cardinality=10)
    result = ohe.fit_transform(sample_df)
    assert any("device_ohe_" in c for c in result.columns)
    # OHE columns should be binary
    for col in [c for c in result.columns if "ohe" in c]:
        assert result[col].isin([0, 1]).all()


# ── Temporal ─────────────────────────────────────────────────────────────────

def test_calendar_features(sample_df):
    cf = CalendarFeatures(columns=["session_start"])
    result = cf.fit_transform(sample_df)
    assert "session_start_month" in result.columns
    assert "session_start_is_weekend" in result.columns


def test_cyclical_encoder(sample_df):
    # First generate calendar features to have _month, _hour, _weekday
    cf = CalendarFeatures(columns=["session_start"])
    cal_df = pd.concat([sample_df, cf.fit_transform(sample_df)], axis=1)
    ce = CyclicalEncoder(columns=["session_start_month", "session_start_hour"])
    result = ce.fit_transform(cal_df)
    assert "session_start_month_sin" in result.columns
    assert "session_start_month_cos" in result.columns
    assert result["session_start_month_sin"].between(-1.0, 1.0).all()


# ── Validation ───────────────────────────────────────────────────────────────

def test_validator_passes(sample_df):
    config = {
        "validation": {
            "required_columns": ["age", "purchased"],
            "max_missing_pct": 0.3,
            "min_rows": 10,
        }
    }
    result = DataValidator(config).validate(sample_df)
    assert result.passed
    assert result.stats["n_rows"] == 200


def test_validator_fails_missing_column(sample_df):
    config = {"validation": {"required_columns": ["nonexistent_col"], "max_missing_pct": 0.3, "min_rows": 10}}
    result = DataValidator(config).validate(sample_df)
    assert not result.passed
    assert any("Missing" in e for e in result.errors)


def test_validator_fails_too_few_rows():
    tiny_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    config = {"validation": {"required_columns": [], "max_missing_pct": 0.3, "min_rows": 100}}
    result = DataValidator(config).validate(tiny_df)
    assert not result.passed
