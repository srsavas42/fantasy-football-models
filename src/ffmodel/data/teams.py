"""Canonical team codes.

Legacy PFR CSVs and nflverse use different abbreviations (GNB vs GB, KAN vs KC,
SFO vs SF ...), and franchises relocate (St. Louis -> LA Rams, San Diego -> LA
Chargers, Oakland -> Las Vegas). Left unnormalized, mixing sources breaks joins,
and — worse — a relocated franchise makes every returning player look like they
changed teams. `normalize_team` maps every known variant to one stable franchise
code so team membership is consistent across seasons and sources.
"""

from __future__ import annotations

import pandas as pd

# variant -> canonical (modern nflverse-style) franchise code.
_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "GNB": "GB", "GB": "GB",
    "HST": "HOU",
    "JAC": "JAX",
    "KAN": "KC", "KC": "KC",
    "NWE": "NE", "NE": "NE",
    "NOR": "NO", "NO": "NO",
    "SFO": "SF", "SF": "SF",
    "TAM": "TB", "TB": "TB",
    # Rams franchise
    "STL": "LAR", "LA": "LAR", "RAM": "LAR", "LAR": "LAR",
    # Chargers franchise
    "SD": "LAC", "SDG": "LAC", "LAC": "LAC",
    # Raiders franchise
    "OAK": "LV", "LVR": "LV", "RAI": "LV", "LV": "LV",
    # Washington through its renames
    "WFT": "WAS", "WSH": "WAS", "WAS": "WAS",
}


def normalize_team(code):
    """Map a team code to its canonical franchise code; unknown/NaN pass through."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return code
    c = str(code).strip().upper()
    return _ALIASES.get(c, c)


def normalize_team_series(s: pd.Series) -> pd.Series:
    up = s.astype("string").str.strip().str.upper()
    return up.map(lambda c: _ALIASES.get(c, c) if c is not pd.NA else c)
