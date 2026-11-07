"""Premium by historical burn: replay every past July-August in the county
through this year's trend, and price the resulting payout distribution."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .contract import ContractTerms, liability_per_acre, payout_per_acre
from .model import FittedModel, expected_yield, index_yield


@dataclass(frozen=True)
class Quote:
    fips5: str
    year: int
    expected_yield: float
    trigger_yield: float
    liability_per_acre: float
    pure_premium_per_acre: float
    premium_per_acre: float
    premium_rate: float  # premium / liability
    payout_probability: float
    payout_1_in_10: float  # 90th percentile payout per acre
    mean_payout_when_paid: float
    total_premium: float
    total_liability: float
    simulated: pd.DataFrame  # weather year -> index yield, payout


def quote(model: FittedModel, panel: pd.DataFrame, fips5: str, year: int, terms: ContractTerms) -> Quote:
    weather = panel.loc[panel["fips5"] == fips5]
    if weather.empty:
        raise KeyError(f"No weather history for county {fips5}")
    exp = expected_yield(model, fips5, year)
    idx = index_yield(model, weather, year, terms.calibrated)
    pay = payout_per_acre(exp, idx, terms)
    pure = float(pay.mean())
    loaded = pure * (1 + config.EXPENSE_LOAD) + config.RISK_LOAD * float(pay.std())
    liab = liability_per_acre(exp, terms)
    sim = pd.DataFrame({"weather_year": weather["year"].to_numpy(), "index_yield": idx, "payout_per_acre": pay})
    return Quote(
        fips5=fips5,
        year=year,
        expected_yield=exp,
        trigger_yield=terms.coverage * exp,
        liability_per_acre=liab,
        pure_premium_per_acre=pure,
        premium_per_acre=loaded,
        premium_rate=loaded / liab if liab > 0 else 0.0,
        payout_probability=float((pay > 0).mean()),
        payout_1_in_10=float(np.quantile(pay, 0.9)),
        mean_payout_when_paid=float(pay[pay > 0].mean()) if (pay > 0).any() else 0.0,
        total_premium=loaded * terms.acres,
        total_liability=liab * terms.acres,
        simulated=sim,
    )
