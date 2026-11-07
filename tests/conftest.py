import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soyindex.data import load_panel  # noqa: E402


@pytest.fixture(scope="session")
def real_panel() -> pd.DataFrame:
    return load_panel()


@pytest.fixture()
def synthetic_panel() -> pd.DataFrame:
    """3 counties x 30 years: yield = trend + 2*(July-Aug rain anomaly) + noise."""
    rng = np.random.default_rng(42)
    rows = []
    for fips, base in [("17001", 40.0), ("17003", 35.0), ("17005", 45.0)]:
        for year in range(1990, 2020):
            pcp07, pcp08 = rng.normal(4, 1.5, 2).clip(0.2)
            rain_anom = pcp07 + pcp08 - 8
            rows.append(
                {
                    "fips5": fips,
                    "county": fips,
                    "year": year,
                    "yield_bu_ac": base + 0.6 * (year - 1990) + 2.0 * rain_anom + rng.normal(0, 0.5),
                    "pcp07": pcp07,
                    "pcp08": pcp08,
                    "tmax07": 86 + rng.normal(0, 2),
                    "tmax08": 85 + rng.normal(0, 2),
                    "zndx07": rain_anom / 2,
                    "zndx08": rain_anom / 2,
                }
            )
    return pd.DataFrame(rows)
