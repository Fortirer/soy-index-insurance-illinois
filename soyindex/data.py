"""Loading the county x year panel and county shapes."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config

REQUIRED_COLUMNS = ("fips5", "county", "year", "yield_bu_ac", *config.INDEX_INPUTS)


class DataError(ValueError):
    """Raised when an input file is missing or malformed."""


def load_panel(path: Path = config.PANEL_CSV) -> pd.DataFrame:
    """Load the panel with FIPS as text (casting to int drops leading zeros)."""
    if not Path(path).exists():
        raise DataError(f"Panel file not found: {path}")
    df = pd.read_csv(path, dtype={"fips5": str, "county_ansi": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Panel is missing columns: {missing}")
    if df.duplicated(["fips5", "year"]).any():
        raise DataError("Panel has duplicate county-year rows")
    keep = ["fips5", "county", "ag_district", "year", "acres_planted", "yield_bu_ac", *config.INDEX_INPUTS]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values(["fips5", "year"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def cached_panel() -> pd.DataFrame:
    return load_panel()


def load_boundaries(path: Path = config.BOUNDARIES_TXT) -> dict[str, list[list[float]]]:
    """Return {fips5: [[lon, lat], ...]} from 'fips|lon,lat lon,lat ...' lines."""
    shapes: dict[str, list[list[float]]] = {}
    for line in Path(path).read_text().splitlines():
        if "|" not in line:
            continue
        fips, coords = line.split("|", 1)
        shapes[fips.strip()] = [[float(v) for v in pt.split(",")] for pt in coords.split()]
    return shapes
