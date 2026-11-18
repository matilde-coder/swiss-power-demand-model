"""Fit three daily demand models on 2020-2023 and evaluate them on 2024-2025.

1. Rule baseline   - "same weekday last week": demand 7 days earlier.
2. Linear model    - weather (heating/cooling degree days, sunshine), calendar
                     (weekday, month, holidays) and a linear time trend.
3. Gradient boosting - same information, but free to learn non-linear effects
                     and interactions.

The test years are never used for fitting or tuning.
Outputs: results/predictions.csv (long format) and results/metrics.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRAIN_END, TEST_START = 2023, 2024
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "daily.csv", parse_dates=["date"])
    df["trend"] = (df["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.25
    df["weekday_num"] = df["weekday"].map({d: i for i, d in enumerate(WEEKDAYS)})
    df["lag7"] = df["demand_gwh"].shift(7)
    return df


def linear_design(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["hdd", "cdd", "sunshine_h", "is_holiday", "is_xmas_break", "trend"]].copy()
    X = X.join(pd.get_dummies(pd.Categorical(df["weekday"], categories=WEEKDAYS),
                              prefix="wd", drop_first=True, dtype=float).set_index(df.index))
    X = X.join(pd.get_dummies(pd.Categorical(df["month"], categories=range(1, 13)),
                              prefix="m", drop_first=True, dtype=float).set_index(df.index))
    return X


GBM_FEATURES = ["temp_c", "sunshine_h", "day_of_year", "weekday_num",
                "is_holiday", "is_xmas_break", "trend"]


def metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = actual - pred
    return {
        "MAE_GWh": err.abs().mean(),
        "MAPE_pct": (err.abs() / actual).mean() * 100,
        "bias_pct": (err / actual).mean() * 100,  # >0: model under-forecasts
        "RMSE_GWh": np.sqrt((err ** 2).mean()),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = load()
    train = df[(df["year"] <= TRAIN_END) & df["lag7"].notna()]
    test = df[df["year"] >= TEST_START]

    preds = {"rule_last_week": test["lag7"]}

    lin = LinearRegression().fit(linear_design(train), train["demand_gwh"])
    preds["linear"] = pd.Series(lin.predict(linear_design(test)), index=test.index)

    gbm = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        categorical_features=[GBM_FEATURES.index("weekday_num")], random_state=0,
    ).fit(train[GBM_FEATURES], train["demand_gwh"])
    preds["gradient_boosting"] = pd.Series(gbm.predict(test[GBM_FEATURES]), index=test.index)

    long = pd.concat(
        [test[["date", "demand_gwh"]].assign(model=name, pred_gwh=p) for name, p in preds.items()],
        ignore_index=True,
    )
    long["error_gwh"] = long["demand_gwh"] - long["pred_gwh"]
    long.to_csv(RESULTS / "predictions.csv", index=False)

    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "split": "test 2024-2025", **metrics(test["demand_gwh"], p)})
    m = pd.DataFrame(rows).round(2)
    m.to_csv(RESULTS / "metrics.csv", index=False)
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
