"""Team-investment score: draft-value curve, recency decay, wiring."""

import numpy as np
import pandas as pd

from ffmodel.features import crossseason as cs
from ffmodel.features import investment as inv


def test_draft_value_monotone_and_bounded():
    vals = [inv.draft_value(p) for p in (1, 10, 32, 64, 128, 256)]
    assert vals[0] == 1.0
    assert all(a > b for a, b in zip(vals, vals[1:]))  # strictly decreasing
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_undrafted_has_zero_value():
    assert inv.draft_value(None) == 0.0
    assert inv.draft_value(float("nan")) == 0.0


def test_investment_decays_with_years_since_draft():
    df = pd.DataFrame({
        "player_name": ["Saquon Barkley"] * 3,
        "position": ["RB"] * 3,
        "season": [2018, 2019, 2021],
    })
    out = inv.add_investment(df, source="legacy")
    inv_by_year = dict(zip(out["season"], out["team_investment"]))
    # A high pick, decaying each year since draft.
    assert inv_by_year[2018] > inv_by_year[2019] > inv_by_year[2021] > 0
    assert out["years_since_draft"].tolist() == [0, 1, 3]


def test_undrafted_player_gets_zero_investment():
    df = pd.DataFrame({
        "player_name": ["Totally Made Up Udfa"],
        "position": ["WR"],
        "season": [2019],
    })
    out = inv.add_investment(df, source="legacy")
    assert out["team_investment"].iloc[0] == 0.0


def test_transitions_expose_team_investment():
    t = cs.build_transitions([2016, 2017, 2018], source="legacy")
    assert "team_investment" in t.columns
    assert (t["team_investment"] >= 0).all() and (t["team_investment"] <= 1).all()
    # Some players carry real draft capital.
    assert (t["team_investment"] > 0).any()
