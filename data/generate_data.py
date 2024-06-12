"""
Generates a realistic synthetic e-commerce clickstream dataset.
Features include numerical, categorical, and temporal columns with a binary purchase target.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_ecommerce_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base_date = pd.Timestamp("2023-01-01")
    timestamps = [
        base_date + pd.Timedelta(seconds=int(s))
        for s in rng.integers(0, 365 * 24 * 3600, n_samples)
    ]

    df = pd.DataFrame({
        "user_id": np.arange(1, n_samples + 1),
        "session_start": timestamps,
        "age": rng.integers(18, 65, n_samples).astype(float),
        "session_duration_sec": rng.exponential(300, n_samples).clip(10, 7200).round(1),
        "page_views": rng.integers(1, 50, n_samples).astype(float),
        "cart_value": rng.exponential(80, n_samples).clip(0, 2000).round(2),
        "previous_purchases": rng.integers(0, 20, n_samples).astype(float),
        "device_type": rng.choice(["mobile", "desktop", "tablet"], n_samples, p=[0.5, 0.4, 0.1]),
        "traffic_source": rng.choice(["organic", "paid", "email", "direct", "social"], n_samples),
        "country": rng.choice(["US", "UK", "DE", "FR", "CA", "AU", "JP"], n_samples),
    })

    # Introduce realistic missingness
    for col, miss_rate in [("age", 0.05), ("cart_value", 0.03), ("page_views", 0.02)]:
        mask = rng.random(n_samples) < miss_rate
        df.loc[mask, col] = np.nan

    # Realistic purchase signal
    purchase_score = (
        0.001 * df["session_duration_sec"].fillna(300)
        + 0.03 * df["page_views"].fillna(5)
        + 0.002 * df["previous_purchases"].fillna(0)
        - 0.0001 * df["cart_value"].fillna(80)
        + (df["traffic_source"] == "email").astype(float) * 0.5
        + (df["traffic_source"] == "paid").astype(float) * 0.3
        + rng.normal(0, 0.5, n_samples)
    )
    df["purchased"] = (purchase_score > purchase_score.median()).astype(int)

    return df


if __name__ == "__main__":
    df = generate_ecommerce_data()
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/ecommerce_dataset.csv", index=False)
    print(f"Generated {len(df)} samples → data/ecommerce_dataset.csv")
    print(f"Purchase rate: {df['purchased'].mean():.2%}")
    print(f"Columns: {list(df.columns)}")
