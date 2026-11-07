### How the product works

1. **Expected yield.** Each county gets its own linear technology trend (yields rise about 0.3–0.85 bu/ac per year). The trend for the crop year is the guarantee base.
2. **Weather index.** A ridge regression predicts the county's yield deviation from trend using only **July–August** NOAA nClimDiv values: rainfall, mean max temperature and the Palmer Z-index. Features include a squared rain term (too much rain also hurts), and heat × dryness (heat mostly hurts when it is dry). A shrunken county multiplier captures the ~7× spread in how sensitive counties are to drought.
3. **Index yield** = expected yield + predicted weather deviation.
4. **Payout** per acre = price × max(0, coverage × expected yield − index yield), capped at the full guarantee.
5. **Premium** = average payout if each season from 1980–2025 repeated this year, plus a 20% expense load and 10% of the payout standard deviation as a risk load.

**Calibrated vs conservative.** The raw prediction regresses toward the mean, so it under-states bad years. The calibrated index rescales it to the spread of real yield swings. It catches more losses and hedges better, but it costs more and pays more often in years with no loss.

### Backtest (2001–2025, 2,317 county-years, strictly out of sample)

At 90% coverage:

| Product | Hit rate on loss years | Payouts with no loss | Payout–loss correlation | Revenue variance cut |
|---|---|---|---|---|
| Weather index (calibrated) | 46% | 44% | 0.51 | 19.5% |
| Weather index (conservative) | 40% | 31% | 0.51 | 11.7% |
| Perfect county index (ceiling) | 100% | 0% | 1.00 | 39.6% |

The perfect county index pays on the county's own realised yield. It is the best any county-level product could do, not a product you could sell. Variance cuts use the fair premium without loads, so they are upper bounds.

### Limits to read before using this

- **Basis risk is large.** The index explains about a third of county yield variation. Losses from pests, disease, planting delays (2019) or unexplained shocks (2003) are not covered.
- **The index runs ~2× rich in the backtest.** 2001–2025 was wetter than 1980–2000, so payouts came to about 0.56× the fair (unloaded) premium, and less against the loaded premium. The burn also uses in-sample index values, which can overstate how closely the index tracks past losses. Pricing on the full 1980–2025 history is the conservative choice, since a 1988-type drought can come back. Re-weighting history should be a deliberate pricing decision.
- **County, not farm.** Even a perfect county index misses farm-level losses.
- **Monthly data.** No daily heat stress, dry-spell length, or frost.
- **Coverage thins after 2018** (102 counties reporting down to ~62), which weakens recent trends.
- **Calibration buys hit rate with false payouts.** Its larger variance cut comes from paying more often, not from predicting better.
- **Model choice saw the test period.** Ridge vs gradient boosting, and the county multiplier, were compared on the 2001–2025 backtest; hyperparameters were set a priori but not tuned inside the window. Treat the headline numbers as somewhat optimistic.
- **The county mix changes by year.** Counties without enough history are dropped, and pooled metrics weigh the years with the most reporting counties.
- **The price is a placeholder.** Use the RMA projected price for the crop year.
- This is a research MVP, not a filed insurance product.
