"""Streamlit dashboard: quote, settle and backtest a soybean weather-index policy.

Run:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from soyindex import config
from soyindex.backtest import run_backtest, summarize
from soyindex.contract import ContractTerms, payout_per_acre
from soyindex.data import cached_panel, load_boundaries
from soyindex.geo import write_geojson
from soyindex.model import FittedModel, fit_model, index_yield
from soyindex.pricing import quote

st.set_page_config(page_title="Soy Index Insurance · Illinois", page_icon="🌱", layout="wide")

PRODUCT_LABELS = {
    "weather_index_calibrated": "Weather index (calibrated)",
    "weather_index": "Weather index (conservative)",
    "area_yield": "Perfect county index (ceiling)",
}


@st.cache_resource(show_spinner="Fitting the weather-yield index…")
def get_model() -> FittedModel:
    return fit_model(cached_panel())


@st.cache_data(show_spinner="Running leak-free backtest…")
def get_backtest(coverage: float) -> pd.DataFrame:
    return run_backtest(cached_panel(), coverages=(coverage,))


@st.cache_data(show_spinner="Pricing every county…")
def state_quotes(year: int, coverage: float, price: float, calibrated: bool) -> pd.DataFrame:
    panel, model = cached_panel(), get_model()
    terms = ContractTerms(coverage=coverage, price_usd_bu=price, acres=1, calibrated=calibrated)
    names = panel.drop_duplicates("fips5").set_index("fips5")["county"]
    rows = []
    for fips in model.trends["fips5"]:
        q = quote(model, panel, fips, year, terms)
        rows.append(
            {
                "fips5": fips,
                "county": names.get(fips, fips).title(),
                "expected_yield": round(q.expected_yield, 1),
                "premium_per_acre": round(q.premium_per_acre, 2),
                "premium_rate_pct": round(100 * q.premium_rate, 2),
                "payout_probability_pct": round(100 * q.payout_probability, 1),
            }
        )
    return pd.DataFrame(rows)


def sidebar() -> tuple[str, int, ContractTerms]:
    panel, model = cached_panel(), get_model()
    counties = (
        panel[panel["fips5"].isin(model.trends["fips5"])]
        .drop_duplicates("fips5")
        .assign(label=lambda d: d["county"].str.title())
        .sort_values("label")
    )
    labels = dict(zip(counties["label"], counties["fips5"]))
    st.sidebar.header("Policy")
    default = list(labels).index("Champaign") if "Champaign" in labels else 0
    county = st.sidebar.selectbox("County", list(labels), index=default)
    next_year = model.last_train_year + 1
    year = st.sidebar.selectbox("Crop year", [next_year, next_year + 1], index=0)
    coverage = st.sidebar.slider("Coverage level", 0.70, 0.95, config.DEFAULT_COVERAGE, 0.05)
    price = st.sidebar.number_input(
        "Price (USD/bu)", 5.0, 25.0, config.DEFAULT_PRICE_USD_BU, 0.25, help="Use the RMA projected price for the year."
    )
    acres = st.sidebar.number_input("Insured acres", 1, 50_000, config.DEFAULT_ACRES, 10)
    calibrated = st.sidebar.toggle(
        "Calibrated index",
        value=True,
        help="Widens the index to match real yield swings: catches more losses, costs more, more false payouts.",
    )
    terms = ContractTerms(coverage=coverage, price_usd_bu=price, acres=acres, calibrated=calibrated)
    return labels[county], int(year), terms


def quote_tab(fips: str, year: int, terms: ContractTerms) -> None:
    q = quote(get_model(), cached_panel(), fips, year, terms)
    c = st.columns(4)
    c[0].metric("Expected yield (bu/ac)", f"{q.expected_yield:.1f}", help="County technology trend for the crop year")
    c[1].metric("Trigger yield (bu/ac)", f"{q.trigger_yield:.1f}")
    c[2].metric("Premium ($/ac)", f"{q.premium_per_acre:.2f}", help=f"Pure premium (average payout) ${q.pure_premium_per_acre:.2f} plus loads")
    c[3].metric("Premium rate", f"{100 * q.premium_rate:.1f}%", help="Premium as a share of the maximum payout")
    c = st.columns(4)
    c[0].metric("Total premium ($)", f"{q.total_premium:,.0f}", help=f"Total liability ${q.total_liability:,.0f}")
    c[1].metric("Chance of a payout", f"{100 * q.payout_probability:.0f}%")
    c[2].metric("Avg payout if paid ($/ac)", f"{q.mean_payout_when_paid:.2f}")
    c[3].metric("Max payout ($/ac)", f"{q.liability_per_acre:.0f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Payout schedule")
        grid = pd.DataFrame({"index_yield": [q.expected_yield * f / 100 for f in range(40, 111)]})
        grid["payout_per_acre"] = payout_per_acre(q.expected_yield, grid["index_yield"], terms)
        line = alt.Chart(grid).mark_line().encode(
            x=alt.X("index_yield:Q", title="Index yield (bu/ac)"),
            y=alt.Y("payout_per_acre:Q", title="Payout (USD/ac)"),
        )
        rule = alt.Chart(pd.DataFrame({"x": [q.trigger_yield]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(line + rule, use_container_width=True)
    with right:
        st.subheader("If a past season repeated this year")
        sim = q.simulated.assign(paid=lambda d: d["payout_per_acre"] > 0)
        bars = alt.Chart(sim).mark_bar().encode(
            x=alt.X("weather_year:O", title="Weather year"),
            y=alt.Y("payout_per_acre:Q", title="Payout (USD/ac)"),
            tooltip=["weather_year", alt.Tooltip("index_yield:Q", format=".1f"), alt.Tooltip("payout_per_acre:Q", format=".2f")],
        )
        st.altair_chart(bars, use_container_width=True)


def settle_tab(fips: str, year: int, terms: ContractTerms) -> None:
    st.markdown(
        "Enter the county's published **NOAA nClimDiv** values for July and August. "
        "Defaults are the most recent season on file."
    )
    panel = cached_panel()
    last = panel[panel["fips5"] == fips].iloc[-1]
    c = st.columns(3)
    vals = {
        "pcp07": c[0].number_input("July rain (in)", 0.0, 25.0, float(last.pcp07), 0.1),
        "pcp08": c[0].number_input("August rain (in)", 0.0, 25.0, float(last.pcp08), 0.1),
        "tmax07": c[1].number_input("July mean max temp (°F)", 60.0, 110.0, float(last.tmax07), 0.1),
        "tmax08": c[1].number_input("August mean max temp (°F)", 60.0, 110.0, float(last.tmax08), 0.1),
        "zndx07": c[2].number_input("July Palmer Z", -8.0, 12.0, float(last.zndx07), 0.1),
        "zndx08": c[2].number_input("August Palmer Z", -8.0, 12.0, float(last.zndx08), 0.1),
    }
    st.caption(f"Defaults from crop year {int(last.year)}.")
    row = pd.DataFrame([{"fips5": fips, "year": year, **vals}])
    model = get_model()
    idx = float(index_yield(model, row, year, terms.calibrated)[0])
    q = quote(model, panel, fips, year, terms)
    pay = float(payout_per_acre(q.expected_yield, idx, terms))
    c = st.columns(3)
    c[0].metric("Index yield (bu/ac)", f"{idx:.1f}", f"{idx - q.expected_yield:+.1f} vs expected")
    c[1].metric("Payout ($/ac)", f"{pay:.2f}")
    c[2].metric("Total payout ($)", f"{pay * terms.acres:,.0f}")


def backtest_tab(fips: str, terms: ContractTerms) -> None:
    st.markdown(
        "Each year from 2001 is priced and settled using **only earlier years**, then compared with the "
        "county's real yield loss. The perfect county index pays on the county's own actual yield: it is the ceiling for any "
        "county-level product, not a sellable product. Hedging metrics use the fair (unloaded) premium."
    )
    bt = get_backtest(terms.coverage)
    s = summarize(bt).assign(product=lambda d: d["product"].map(PRODUCT_LABELS))
    st.dataframe(
        s.drop(columns=["coverage"]).set_index("product").style.format(precision=3),
        use_container_width=True,
    )
    product = "weather_index_calibrated" if terms.calibrated else "weather_index"
    g = bt[(bt["fips5"] == fips) & (bt["product"] == product)]
    long = pd.concat(
        [
            g[["year"]].assign(kind="Actual county loss", usd=g["actual_shortfall"] * terms.price_usd_bu),
            g[["year"]].assign(kind="Index payout", usd=g["index_shortfall"] * terms.price_usd_bu),
        ]
    )
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("year:O", title=None),
        xOffset="kind:N",
        y=alt.Y("usd:Q", title="USD/acre"),
        color=alt.Color("kind:N", title=None),
        tooltip=["year", "kind", alt.Tooltip("usd:Q", format=".2f")],
    )
    st.subheader("This county, year by year")
    st.altair_chart(chart, use_container_width=True)


def map_tab(year: int, terms: ContractTerms) -> None:
    df = state_quotes(year, terms.coverage, terms.price_usd_bu, terms.calibrated)
    geo_file = config.ROOT / "static" / "il_counties.geojson"
    if not geo_file.exists():
        write_geojson(load_boundaries(), geo_file)
    cols = ["county", "expected_yield", "premium_per_acre", "premium_rate_pct", "payout_probability_pct"]
    chart = (
        # Streamlit flattens inline data, so shapes come from a static file and numbers are joined on.
        alt.Chart(alt.UrlData("app/static/il_counties.geojson", format=alt.DataFormat(property="features", type="json")))
        .mark_geoshape(stroke="white", strokeWidth=0.5)
        .transform_lookup(lookup="properties.fips5", from_=alt.LookupData(df, "fips5", cols))
        .encode(
            color=alt.Color("premium_rate_pct:Q", title="Premium rate %", scale=alt.Scale(scheme="orangered")),
            tooltip=[
                alt.Tooltip("county:N", title="County"),
                alt.Tooltip("expected_yield:Q", title="Expected yield"),
                alt.Tooltip("premium_per_acre:Q", title="Premium $/ac"),
                alt.Tooltip("premium_rate_pct:Q", title="Rate %"),
                alt.Tooltip("payout_probability_pct:Q", title="Payout chance %"),
            ],
        )
        .project("mercator")
        .properties(height=620)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df.drop(columns="fips5").sort_values("premium_rate_pct", ascending=False), hide_index=True)


def main() -> None:
    st.title("🌱 Soybean weather-index insurance · Illinois")
    st.caption(
        "Pays when July–August weather points to a county yield below the trigger. No loss adjustment: "
        "settlement uses six public NOAA values. Built on the soybean-climate-illinois panel (NASS + NOAA, 1980–2025)."
    )
    fips, year, terms = sidebar()
    tabs = st.tabs(["Quote", "Settle a season", "Backtest", "State map", "Method & limits"])
    with tabs[0]:
        quote_tab(fips, year, terms)
    with tabs[1]:
        settle_tab(fips, year, terms)
    with tabs[2]:
        backtest_tab(fips, terms)
    with tabs[3]:
        map_tab(year, terms)
    with tabs[4]:
        st.markdown((config.ROOT / "docs" / "METHOD.md").read_text(encoding="utf-8"))


main()
