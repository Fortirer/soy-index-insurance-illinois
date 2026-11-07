"""Central settings for the soybean index-insurance MVP."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

PANEL_CSV = DATA_DIR / "soybean_illinois_climate_1980_2025.csv"
BOUNDARIES_TXT = DATA_DIR / "il_county_boundaries.txt"

# The six public NOAA nClimDiv monthly values the index is settled on.
# Keeping the index to these makes settlement reproducible by anyone.
INDEX_INPUTS = ("pcp07", "pcp08", "tmax07", "tmax08", "zndx07", "zndx08")

# Trend: a county needs this many prior years before we trust its trend line.
MIN_TREND_YEARS = 10

# Model
RIDGE_ALPHA = 10.0
# Shrinkage strength for county sensitivity (in "pseudo-years" pulled toward 1.0).
SENSITIVITY_SHRINK_YEARS = 15.0
SENSITIVITY_BOUNDS = (0.3, 3.0)
CALIBRATION_MAX_SCALE = 2.5

# Backtest
BACKTEST_START_YEAR = 2001

# Contract defaults (all user-adjustable in the app)
DEFAULT_COVERAGE = 0.90
DEFAULT_PRICE_USD_BU = 10.50  # assumption; replace with the RMA projected price
DEFAULT_ACRES = 500
EXPENSE_LOAD = 0.20  # admin + distribution, as a share of pure premium
RISK_LOAD = 0.10  # share of payout standard deviation added for capital cost
