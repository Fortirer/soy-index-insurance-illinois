# Soybean Weather-Index Insurance · Illinois (MVP)

A working MVP for **county-level weather-index insurance for Illinois soybeans**. It quotes a premium, settles a season from six public NOAA numbers, and backtests the product out of sample against real county yield losses.

Built on the [`soybean-climate-illinois`](https://github.com/osperry/soybean-climate-illinois) panel by Oric Perry (MIT): USDA NASS county yields and NOAA nClimDiv climate, 102 counties, 1980–2025. Its findings set the design: July–August is the window that matters, heat hurts mainly when it is dry, and county sensitivity varies about 7×.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py              # dashboard
python scripts/run_backtest.py    # writes results/backtest_summary.csv
python -m pytest                  # 25 tests
```

## What's inside

| Path | Purpose |
|---|---|
| `soyindex/trend.py` | Per-county technology trend → expected yield |
| `soyindex/features.py` | July–August features from 6 NOAA values |
| `soyindex/model.py` | Ridge index + shrunken county sensitivity + optional calibration |
| `soyindex/contract.py` | Contract terms and payout formula |
| `soyindex/pricing.py` | Burn-cost premium with expense and risk loads |
| `soyindex/backtest.py` | Expanding-window backtest and basis-risk metrics |
| `app.py` | Streamlit dashboard: Quote · Settle · Backtest · Map · Method |
| `docs/METHOD.md` | Method, backtest results and limits |

## Payout

```
payout/acre = price × max(0, coverage × expected_yield − index_yield)
index_yield = county trend(year) + f(July–Aug rain, max temp, Palmer Z)
```

## Results at 90% coverage (2001–2025, out of sample)

| Product | Hit rate | No-loss payouts | Variance cut |
|---|---|---|---|
| Weather index (calibrated, default) | 46% | 44% | 19.5% |
| Weather index (conservative) | 40% | 31% | 11.7% |
| Perfect county index (ceiling) | 100% | 0% | 39.6% |

See `docs/METHOD.md` for limits. In short: basis risk is large, and the index ran about 2× rich over 2001–2025 because the weather was milder than in the 1980s. This is a research MVP, not a filed insurance product.
