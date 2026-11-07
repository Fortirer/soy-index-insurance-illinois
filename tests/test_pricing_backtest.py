import pytest

from soyindex.backtest import PRODUCTS, run_backtest, summarize
from soyindex.contract import ContractTerms
from soyindex.model import fit_model
from soyindex.pricing import quote


@pytest.fixture(scope="module")
def real_model(real_panel):
    return fit_model(real_panel)


def test_quote_is_internally_consistent(real_model, real_panel):
    terms = ContractTerms(coverage=0.9, price_usd_bu=10.5, acres=500)
    q = quote(real_model, real_panel, "17019", 2026, terms)  # Champaign
    assert 55 < q.expected_yield < 80
    assert q.premium_per_acre >= q.pure_premium_per_acre > 0
    assert 0 < q.premium_rate < 0.5
    assert 0 <= q.payout_probability <= 1
    assert q.total_premium == pytest.approx(q.premium_per_acre * 500)
    assert len(q.simulated) == (real_panel.fips5 == "17019").sum()


def test_higher_coverage_costs_more(real_model, real_panel):
    low = quote(real_model, real_panel, "17019", 2026, ContractTerms(coverage=0.75))
    high = quote(real_model, real_panel, "17019", 2026, ContractTerms(coverage=0.90))
    assert high.premium_per_acre > low.premium_per_acre


def test_quote_unknown_county_raises(real_model, real_panel):
    with pytest.raises(KeyError):
        quote(real_model, real_panel, "17999", 2026, ContractTerms())


def test_backtest_is_out_of_sample_and_summarizes(real_panel):
    bt = run_backtest(real_panel, coverages=(0.9,), start_year=2022)
    assert bt["year"].min() == 2022
    assert set(bt["product"]) == set(PRODUCTS)
    s = summarize(bt).set_index("product")
    # Area-yield pays exactly when the county had a loss, by construction.
    assert s.loc["area_yield", "hit_rate"] == 1.0
    assert s.loc["area_yield", "false_payout_share"] == 0.0


def test_backtest_premium_ignores_the_test_year(real_panel):
    """Changing 2025 yields must not change the 2025 premium (no leakage)."""
    shocked = real_panel.copy()
    shocked.loc[shocked.year == 2025, "yield_bu_ac"] = 1.0
    a = run_backtest(real_panel, coverages=(0.9,), start_year=2025)
    b = run_backtest(shocked, coverages=(0.9,), start_year=2025)
    assert a["fair_premium_bu"].tolist() == pytest.approx(b["fair_premium_bu"].tolist())
    # Weather products settle on weather only; area_yield settles on yield by design.
    weather_a = a[a["product"] != "area_yield"]
    weather_b = b[b["product"] != "area_yield"]
    assert weather_a["index_shortfall"].tolist() == pytest.approx(weather_b["index_shortfall"].tolist())


def test_backtest_rejects_start_year_without_training_history(real_panel):
    with pytest.raises(ValueError, match="start_year"):
        run_backtest(real_panel, start_year=1985)


def test_fit_model_needs_counties_with_enough_history(real_panel):
    with pytest.raises(ValueError, match="history"):
        fit_model(real_panel[real_panel.year < 1985])
