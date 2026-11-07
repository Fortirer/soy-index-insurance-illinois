import json

import pandas as pd

from soyindex.geo import _signed_area, to_features, write_geojson


def test_rings_are_closed_and_clockwise():
    ccw = [[0, 0], [1, 0], [1, 1], [0, 1]]
    feat = to_features({"17001": ccw}, pd.DataFrame({"fips5": ["17001"], "v": [1.0]}))[0]
    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert _signed_area(ring) < 0
    assert feat["properties"]["v"] == 1.0


def test_unknown_county_is_skipped():
    assert to_features({}, pd.DataFrame({"fips5": ["17999"]})) == []


def test_write_geojson_round_trips(tmp_path):
    path = write_geojson({"17001": [[0, 0], [0, 1], [1, 1]]}, tmp_path / "x.geojson")
    data = json.loads(path.read_text())
    assert data["type"] == "FeatureCollection"
    assert data["features"][0]["properties"]["fips5"] == "17001"
