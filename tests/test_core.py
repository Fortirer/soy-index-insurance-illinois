import numpy as np
import pandas as pd
import pytest

from soyindex.contract import ContractTerms, liability_per_acre, payout_per_acre
from soyindex.data import DataError, load_boundaries, load_panel
from soyindex.features import build_features, county_normals, season_summary
from soyindex.model import expected_yield, fit_model, index_yield, predict_anomaly
from soyindex.trend import add_trend, fit_county_trends

# ---------- data ----------


def test_panel_keeps_fips_as_text_with_leading_zeros(real_panel):
    assert real_panel["fips5"].dtype == object
    assert "17001" in set(real_panel["fips5"])
    assert real_panel["fips5"].str.len().eq(5).all()


def test_missing_panel_raises_data_error(tmp_path):
    with pytest.raises(DataError):
        load_panel(tmp_path / "nope.csv")


def test_panel_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"fips5": ["17001"], "year": [2000]}).to_csv(p, index=False)
    with pytest.raises(DataError, match="missing columns"):
        load_panel(p)


def test_boundaries_parse_into_lon_lat_rings():
    shapes = load_boundaries()
    assert len(shapes) >= 100
    lon, lat = shapes["17019"][0]
    assert -92 < lon < -87 and 36 < lat < 43


# ---------- trend ----------


def test_trend_recovers_known_slope():
    df = pd.DataFrame({"fips5": "17001", "year": range(2000, 2020), "yield_bu_ac": [30 + 0.5 * i for i in range(20)]})
    t = fit_county_trends(df)
    assert t["slope"].iloc[0] == pytest.approx(0.5)


def test_trend_skips_counties_with_short_history():
    df = pd.DataFrame({"fips5": "17001", "year": range(2000, 2005), "yield_bu_ac": 40.0})
    assert fit_county_trends(df, min_years=10).empty


def test_add_trend_for_fixed_year_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    t = fit_county_trends(synthetic_panel)
    out = add_trend(synthetic_panel, t, year=2030)
    pd.testing.assert_frame_equal(synthetic_panel, before)
    assert out.groupby("fips5")["trend_yield"].nunique().eq(1).all()


# ---------- features ----------


def test_season_summary_matches_definitions(synthetic_panel):
    s = season_summary(synthetic_panel)
    row = synthetic_panel.iloc[0]
    assert s["pcp_ja"].iloc[0] == pytest.approx(row.pcp07 + row.pcp08)
    assert s["tmax_ja"].iloc[0] == pytest.approx((row.tmax07 + row.tmax08) / 2)


def test_heat_penalty_only_applies_when_dry(synthetic_panel):
    f = build_features(synthetic_panel, county_normals(synthetic_panel))
    wet = season_summary(synthetic_panel)["zndx_ja"] >= 0
    assert (f.loc[wet, "heat_x_dry"] == 0).all()


def test_unknown_county_falls_back_to_state_normals(synthetic_panel):
    normals = county_normals(synthetic_panel)
    row = synthetic_panel.iloc[[0]].assign(fips5="17999")
    f = build_features(row, normals)
    assert f["normal_pcp_ja"].iloc[0] == pytest.approx(normals["normal_pcp_ja"].mean())


# ---------- contract ----------


def test_no_payout_above_trigger():
    terms = ContractTerms(coverage=0.9, price_usd_bu=10, acres=100)
    assert payout_per_acre(50, 46, terms) == 0  # trigger is 45


def test_payout_is_linear_below_trigger_and_capped():
    terms = ContractTerms(coverage=0.9, price_usd_bu=10, acres=100)
    assert payout_per_acre(50, 40, terms) == pytest.approx(50)  # (45-40)*10
    assert payout_per_acre(50, -10, terms) == pytest.approx(liability_per_acre(50, terms))


@pytest.mark.parametrize("kwargs", [{"coverage": 0.99}, {"price_usd_bu": 0}, {"acres": -1}])
def test_invalid_terms_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ContractTerms(**kwargs)


# ---------- model ----------


def test_model_learns_that_dry_seasons_cut_yield(synthetic_panel):
    m = fit_model(synthetic_panel)
    dry = synthetic_panel.iloc[[0]].assign(pcp07=1.0, pcp08=1.0, zndx07=-3.0, zndx08=-3.0)
    wet = synthetic_panel.iloc[[0]].assign(pcp07=5.0, pcp08=5.0, zndx07=1.0, zndx08=1.0)
    assert predict_anomaly(m, dry)[0] < 0 < predict_anomaly(m, wet)[0]


def test_calibration_widens_the_index(synthetic_panel):
    m = fit_model(synthetic_panel)
    raw = predict_anomaly(m, synthetic_panel)
    cal = predict_anomaly(m, synthetic_panel, calibrated=True)
    assert np.std(cal) >= np.std(raw)


def test_index_yield_uses_requested_year_trend(synthetic_panel):
    m = fit_model(synthetic_panel)
    rows = synthetic_panel[synthetic_panel.fips5 == "17001"]
    idx = index_yield(m, rows, 2025)
    anom = predict_anomaly(m, rows.reset_index(drop=True))
    assert idx == pytest.approx(expected_yield(m, "17001", 2025) + anom)


def test_fit_on_empty_raises():
    with pytest.raises(ValueError):
        fit_model(pd.DataFrame())


def test_expected_yield_unknown_county_raises(synthetic_panel):
    with pytest.raises(KeyError):
        expected_yield(fit_model(synthetic_panel), "17999", 2025)


def test_non_positive_expected_yield_never_pays():
    terms = ContractTerms()
    assert payout_per_acre(-5.0, 10.0, terms) == 0
    assert liability_per_acre(-5.0, terms) == 0
