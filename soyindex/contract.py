"""Contract terms and the payout formula.

Payout per acre = price x max(0, coverage x expected_yield - index_yield),
capped at the full guarantee. The farmer's own harvest is never measured:
settlement uses only the published index.
"""

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class ContractTerms:
    coverage: float = config.DEFAULT_COVERAGE
    price_usd_bu: float = config.DEFAULT_PRICE_USD_BU
    acres: float = config.DEFAULT_ACRES
    calibrated: bool = True

    def __post_init__(self) -> None:
        if not 0.5 <= self.coverage <= 0.95:
            raise ValueError("coverage must be between 0.50 and 0.95")
        if self.price_usd_bu <= 0:
            raise ValueError("price must be positive")
        if self.acres <= 0:
            raise ValueError("acres must be positive")


def trigger_yield(expected: float, terms: ContractTerms) -> float:
    return terms.coverage * expected


def payout_per_acre(expected: float, index_yields: np.ndarray | float, terms: ContractTerms) -> np.ndarray:
    """Vectorized payout in USD/acre for one or many index outcomes."""
    trigger = max(trigger_yield(expected, terms), 0.0)
    shortfall = np.clip(trigger - np.asarray(index_yields, dtype=float), 0.0, trigger)
    return shortfall * terms.price_usd_bu


def liability_per_acre(expected: float, terms: ContractTerms) -> float:
    """Maximum possible payout per acre."""
    return max(trigger_yield(expected, terms), 0.0) * terms.price_usd_bu
