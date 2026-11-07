"""Index features, built only from the six public July-August NOAA values.

Design follows the soybean-climate-illinois findings:
- July-August (pod set, seed fill) is the window that matters.
- Rain response is nonlinear with an interior optimum -> squared term.
- Heat hurts mostly when it is dry -> heat x dryness interaction.
"""

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "pcp_ja",
    "pcp_ja_sq",
    "pcp_ja_anom",
    "tmax_ja",
    "tmax_ja_anom",
    "zndx_ja",
    "heat_x_dry",
    "pcp08",
    "tmax08",
    "normal_pcp_ja",
    "normal_tmax_ja",
)


def season_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse monthly inputs into July-August totals/means."""
    return pd.DataFrame(
        {
            "pcp_ja": rows["pcp07"] + rows["pcp08"],
            "tmax_ja": (rows["tmax07"] + rows["tmax08"]) / 2,
            "zndx_ja": (rows["zndx07"] + rows["zndx08"]) / 2,
        },
        index=rows.index,
    )


def county_normals(train: pd.DataFrame) -> pd.DataFrame:
    """Per-county mean July-August rain and max temperature over the training years."""
    s = season_summary(train).assign(fips5=train["fips5"].to_numpy())
    return (
        s.groupby("fips5")
        .agg(normal_pcp_ja=("pcp_ja", "mean"), normal_tmax_ja=("tmax_ja", "mean"))
        .reset_index()
    )


def build_features(rows: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix aligned to `rows.index`. Counties without normals fall back
    to the statewide mean normal so a quote never silently fails."""
    s = season_summary(rows)
    n = rows[["fips5"]].merge(normals, on="fips5", how="left").set_index(rows.index)
    n = n.fillna({"normal_pcp_ja": normals["normal_pcp_ja"].mean(), "normal_tmax_ja": normals["normal_tmax_ja"].mean()})
    tmax_anom = s["tmax_ja"] - n["normal_tmax_ja"]
    dryness = np.clip(-s["zndx_ja"], 0, None)
    return pd.DataFrame(
        {
            "pcp_ja": s["pcp_ja"],
            "pcp_ja_sq": s["pcp_ja"] ** 2,
            "pcp_ja_anom": s["pcp_ja"] - n["normal_pcp_ja"],
            "tmax_ja": s["tmax_ja"],
            "tmax_ja_anom": tmax_anom,
            "zndx_ja": s["zndx_ja"],
            "heat_x_dry": tmax_anom * dryness,
            "pcp08": rows["pcp08"],
            "tmax08": rows["tmax08"],
            "normal_pcp_ja": n["normal_pcp_ja"],
            "normal_tmax_ja": n["normal_tmax_ja"],
        },
        index=rows.index,
    )[list(FEATURE_NAMES)]
