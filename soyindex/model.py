"""Weather yield index: predicts a county's yield shortfall from July-August weather.

Pooled ridge regression on detrended yield, then a per-county sensitivity
multiplier (counties differ ~7x in how much a dry August hurts them), shrunk
toward 1.0 so short histories do not produce wild multipliers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import build_features, county_normals
from .trend import add_trend, fit_county_trends


@dataclass(frozen=True)
class FittedModel:
    trends: pd.DataFrame
    normals: pd.DataFrame
    regressor: Pipeline
    sensitivity: pd.Series  # indexed by fips5
    calibration_scale: float
    last_train_year: int


def _county_sensitivity(fips: pd.Series, pred: np.ndarray, anom: np.ndarray) -> pd.Series:
    df = pd.DataFrame({"fips5": fips.to_numpy(), "p": pred, "a": anom})
    k = config.SENSITIVITY_SHRINK_YEARS
    lo, hi = config.SENSITIVITY_BOUNDS
    out = {}
    for f, g in df.groupby("fips5"):
        var = float(np.var(g["p"]))
        raw = float(np.cov(g["p"], g["a"], bias=True)[0, 1] / var) if var > 1e-9 else 1.0
        n = len(g)
        out[f] = float(np.clip((n * raw + k * 1.0) / (n + k), lo, hi))
    return pd.Series(out, name="sensitivity")


def fit_model(train: pd.DataFrame) -> FittedModel:
    """Fit trends, normals, ridge and county sensitivities on `train` only."""
    if train.empty:
        raise ValueError("Cannot fit on an empty training set")
    trends = fit_county_trends(train)
    if trends.empty:
        raise ValueError(f"No county has {config.MIN_TREND_YEARS}+ years of history to fit a trend")
    rows = add_trend(train, trends)
    normals = county_normals(train)
    X = build_features(rows, normals)
    y = rows["yield_anom"].to_numpy()
    reg = make_pipeline(StandardScaler(), Ridge(alpha=config.RIDGE_ALPHA)).fit(X, y)
    pooled = reg.predict(X)
    sens = _county_sensitivity(rows["fips5"], pooled, y)
    adjusted = pooled * rows["fips5"].map(sens).fillna(1.0).to_numpy()
    sd_pred = float(np.std(adjusted))
    scale = min(float(np.std(y)) / sd_pred, config.CALIBRATION_MAX_SCALE) if sd_pred > 0 else 1.0
    return FittedModel(trends, normals, reg, sens, scale, int(train["year"].max()))


def predict_anomaly(model: FittedModel, rows: pd.DataFrame, calibrated: bool = False) -> np.ndarray:
    """Weather-driven yield anomaly (bu/ac) for each row's county and weather."""
    X = build_features(rows, model.normals)
    pred = model.regressor.predict(X) * rows["fips5"].map(model.sensitivity).fillna(1.0).to_numpy()
    return pred * model.calibration_scale if calibrated else pred


def expected_yield(model: FittedModel, fips5: str, year: int) -> float:
    """Trend yield for a county in a given year (the contract's guarantee base)."""
    t = model.trends.loc[model.trends["fips5"] == fips5]
    if t.empty:
        raise KeyError(f"No trend for county {fips5}")
    return float(t["intercept"].iloc[0] + t["slope"].iloc[0] * year)


def index_yield(model: FittedModel, rows: pd.DataFrame, year: int, calibrated: bool = False) -> np.ndarray:
    """Index yield = county trend for `year` + weather anomaly from each row's weather."""
    with_trend = add_trend(rows.reset_index(drop=True), model.trends, year=year)
    return with_trend["trend_yield"].to_numpy() + predict_anomaly(model, with_trend, calibrated)
