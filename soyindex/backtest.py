"""Leak-free expanding-window backtest.

For each year Y, everything (trends, normals, model, premium) is fitted on
years < Y only, then the contract is settled on year Y's real weather and
compared against the county's real yield loss.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config
from .model import fit_model, index_yield
from .trend import add_trend

PRODUCTS = ("weather_index", "weather_index_calibrated", "area_yield")


def _burn_premium(train_rows: pd.DataFrame, idx: np.ndarray, exp_by_fips: pd.Series, coverage: float) -> pd.Series:
    """Fair (pure) premium per county per unit price: mean past shortfall."""
    exp = train_rows["fips5"].map(exp_by_fips).to_numpy()
    trigger = coverage * exp
    short = np.clip(trigger - idx, 0, trigger)
    return pd.Series(short, index=train_rows["fips5"].to_numpy()).groupby(level=0).mean()


def run_backtest(
    panel: pd.DataFrame,
    coverages: Iterable[float] = (0.80, 0.85, 0.90),
    start_year: int = config.BACKTEST_START_YEAR,
) -> pd.DataFrame:
    """Per county-year-coverage-product: expected, actual, index yield, shortfalls (bu/ac)."""
    if start_year <= int(panel["year"].min()) + config.MIN_TREND_YEARS:
        raise ValueError(f"start_year must leave at least {config.MIN_TREND_YEARS} years of training data")
    out = []
    for year in range(start_year, int(panel["year"].max()) + 1):
        train = panel[panel["year"] < year]
        test = panel[panel["year"] == year]
        if test.empty:
            continue
        model = fit_model(train)
        test = add_trend(test, model.trends)
        if test.empty:
            continue
        exp_by_fips = test.set_index("fips5")["trend_yield"]
        train_c = train[train["fips5"].isin(exp_by_fips.index)].reset_index(drop=True)
        train_anom = add_trend(train_c, model.trends)  # in-sample anomalies for area-yield burn

        test_idx = {
            "weather_index": index_yield(model, test, year, calibrated=False),
            "weather_index_calibrated": index_yield(model, test, year, calibrated=True),
            "area_yield": test["yield_bu_ac"].to_numpy(),
        }
        burn_idx = {
            "weather_index": index_yield(model, train_c, year, calibrated=False),
            "weather_index_calibrated": index_yield(model, train_c, year, calibrated=True),
            "area_yield": train_anom["fips5"].map(exp_by_fips).to_numpy() + train_anom["yield_anom"].to_numpy(),
        }
        burn_rows = {"area_yield": train_anom}
        for cov in coverages:
            trigger = cov * test["trend_yield"].to_numpy()
            actual_short = np.clip(trigger - test["yield_bu_ac"].to_numpy(), 0, trigger)
            for product in PRODUCTS:
                prem = _burn_premium(burn_rows.get(product, train_c), burn_idx[product], exp_by_fips, cov)
                out.append(
                    pd.DataFrame(
                        {
                            "year": year,
                            "fips5": test["fips5"].to_numpy(),
                            "coverage": cov,
                            "product": product,
                            "expected_yield": test["trend_yield"].to_numpy(),
                            "actual_yield": test["yield_bu_ac"].to_numpy(),
                            "index_yield": test_idx[product],
                            "actual_shortfall": actual_short,
                            "index_shortfall": np.clip(trigger - test_idx[product], 0, trigger),
                            "fair_premium_bu": test["fips5"].map(prem).fillna(0).to_numpy(),
                        }
                    )
                )
    if not out:
        raise ValueError("Backtest produced no county-years; check start_year and the panel")
    return pd.concat(out, ignore_index=True)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation, or NaN when either side never varies (e.g. no payouts)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cvar(x: np.ndarray, q: float = 0.05) -> float:
    cut = np.quantile(x, q)
    return float(x[x <= cut].mean())


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """Basis-risk and hedging metrics per coverage x product.

    Premiums here are fair (unloaded) burn premiums, so hedging gains are an
    upper bound for the loaded product. Revenue is normalized by expected revenue so counties are comparable:
    r = actual/expected (+ index shortfall - premium)/expected.
    """
    rows = []
    for (cov, product), g in bt.groupby(["coverage", "product"]):
        exp = g["expected_yield"].to_numpy()
        base = g["actual_yield"].to_numpy() / exp
        hedged = base + (g["index_shortfall"].to_numpy() - g["fair_premium_bu"].to_numpy()) / exp
        had_loss = g["actual_shortfall"] > 0
        paid = g["index_shortfall"] > 0
        rows.append(
            {
                "coverage": cov,
                "product": product,
                "n_county_years": len(g),
                "loss_events": int(had_loss.sum()),
                "hit_rate": float((had_loss & paid).sum() / max(had_loss.sum(), 1)),
                "false_payout_share": float((~had_loss & paid).sum() / max(paid.sum(), 1)),
                "payout_loss_corr": _safe_corr(g["actual_shortfall"].to_numpy(), g["index_shortfall"].to_numpy()),
                "payout_to_fair_premium": float(g["index_shortfall"].sum() / max(g["fair_premium_bu"].sum(), 1e-9)),
                "variance_reduction": float(1 - np.var(hedged) / np.var(base)),
                "cvar5_unhedged": _cvar(base),
                "cvar5_hedged": _cvar(hedged),
            }
        )
    return pd.DataFrame(rows)

