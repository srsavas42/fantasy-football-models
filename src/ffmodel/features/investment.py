"""Team investment — how much organizational capital a team has in a player.

Teams manufacture opportunity to justify what they spent: a high draft pick or
big guaranteed contract buys a player touches through a slow start that a warm
body would never get. This encodes that commitment as a per-player score, whose
value is highest exactly where usage history is thin (rookies, year-2 players,
committees) and which decays toward zero as real usage accumulates and takes
over the job of describing the role.

v1 is **draft-capital only**, which works offline from the committed combine
draft data (`features/draft.py`). Contract / dead-cap (nflverse
`import_contracts`) is a documented enhancement that activates locally; trade-up
capital has no clean feed and is left for later.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.features.draft import load_draft_capital

# Exponential pick-value curve (normalized to 1.0 at pick 1). A stand-in for the
# nflverse draft-value chart (import_draft_values); swap in that chart when
# online. pick1=1.00, pick16=0.71, pick32=0.49, pick64=0.24, pick100=0.12.
_PICK_SCALE = 45.0
# Draft capital's hold on *role* fades year over year as usage takes over.
_RECENCY_TAU = 4.0


def draft_value(overall_pick) -> float:
    """Normalized draft capital in [0, 1]; undrafted / unknown pick -> 0."""
    if overall_pick is None or pd.isna(overall_pick):
        return 0.0
    return math.exp(-(float(overall_pick) - 1.0) / _PICK_SCALE)


def player_draft_capital(draft_years: Iterable[int], source: str = "auto") -> pd.DataFrame:
    """Per (player_name, position): draft year, overall pick, and draft value.

    Keyed by name+position (the offline identity), keeping the earliest draft
    row per player. Upgraded to gsis ids when `crossseason` adopts the id map.
    """
    dc = load_draft_capital(draft_years, source=source)
    if dc.empty:
        return pd.DataFrame(
            columns=["player_name", "position", "draft_year", "overall_pick", "draft_value"]
        )
    dc = dc.sort_values("season").drop_duplicates(["player_name", "position"], keep="first")
    dc = dc.rename(columns={"season": "draft_year"})
    dc["draft_value"] = dc["overall_pick"].map(draft_value)
    return dc[["player_name", "position", "draft_year", "overall_pick", "draft_value"]]


def add_investment(
    df: pd.DataFrame,
    source: str = "auto",
    draft_years: Iterable[int] | None = None,
    recency_tau: float = _RECENCY_TAU,
) -> pd.DataFrame:
    """Attach `draft_value`, `years_since_draft`, and `team_investment` to a
    frame carrying `player_name`, `position`, and `season` (the as-of year).

    team_investment = draft_value * exp(-years_since_draft / tau). Contract /
    dead-cap adds in here when that feed is wired; absent, it's draft-only.
    """
    out = df.copy()
    if draft_years is None:
        smax = int(pd.to_numeric(out["season"], errors="coerce").max())
        draft_years = range(2000, smax + 1)
    cap = player_draft_capital(draft_years, source=source)

    out = out.merge(cap, on=["player_name", "position"], how="left")
    out["draft_value"] = out["draft_value"].fillna(0.0)
    years_since = pd.to_numeric(out["season"], errors="coerce") - out["draft_year"]
    out["years_since_draft"] = years_since
    decay = np.exp(-years_since.clip(lower=0).fillna(0.0) / recency_tau)
    # No draft capital (undrafted / pre-2000 / missing) -> zero investment.
    out["team_investment"] = np.where(out["draft_value"] > 0, out["draft_value"] * decay, 0.0)
    return out.drop(columns=["draft_year"])
