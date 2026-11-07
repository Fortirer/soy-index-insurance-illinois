"""GeoJSON for the county map (d3/Vega need clockwise exterior rings)."""

import json
from pathlib import Path

import pandas as pd


def _signed_area(ring: list[list[float]]) -> float:
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])) / 2


def _clockwise(ring: list[list[float]]) -> list[list[float]]:
    closed = ring if ring[0] == ring[-1] else [*ring, ring[0]]
    return closed if _signed_area(closed) < 0 else closed[::-1]


def to_features(shapes: dict[str, list[list[float]]], values: pd.DataFrame) -> list[dict]:
    """One GeoJSON feature per county in `values` (needs a `fips5` column)."""
    feats = []
    for rec in values.to_dict("records"):
        ring = shapes.get(rec["fips5"])
        if not ring:
            continue
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [_clockwise(ring)]},
                "properties": rec,
            }
        )
    return feats


def write_geojson(shapes: dict[str, list[list[float]]], path: Path) -> Path:
    """Write county outlines (fips5 only) as a FeatureCollection file."""
    feats = to_features(shapes, pd.DataFrame({"fips5": list(shapes)}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return path
