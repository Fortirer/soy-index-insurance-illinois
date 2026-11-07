"""County technology trends. Yield gains ~0.6 bu/ac/yr dwarf any weather signal,
so every county is detrended on its own line, fitted only on the years given."""

import numpy as np
import pandas as pd

from . import config


def fit_county_trends(panel: pd.DataFrame, min_years: int = config.MIN_TREND_YEARS) -> pd.DataFrame:
    """OLS yield ~ year per county. Returns columns fips5, intercept, slope, n_years."""
    records = []
    for fips, g in panel.groupby("fips5"):
        if len(g) < min_years:
            continue
        slope, intercept = np.polyfit(g["year"].to_numpy(float), g["yield_bu_ac"].to_numpy(float), 1)
        records.append({"fips5": fips, "intercept": intercept, "slope": slope, "n_years": len(g)})
    return pd.DataFrame(records, columns=["fips5", "intercept", "slope", "n_years"])


def add_trend(rows: pd.DataFrame, trends: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Return a copy of rows with `trend_yield` (for `year`, or each row's own year)
    and `yield_anom` when actual yield is present. Rows without a trend are dropped."""
    out = rows.merge(trends[["fips5", "intercept", "slope"]], on="fips5", how="inner")
    yr = out["year"] if year is None else year
    out = out.assign(trend_yield=out["intercept"] + out["slope"] * yr)
    if "yield_bu_ac" in out.columns:
        out = out.assign(yield_anom=out["yield_bu_ac"] - out["trend_yield"])
    return out.drop(columns=["intercept", "slope"])
