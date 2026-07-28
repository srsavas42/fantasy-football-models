"""Dirichlet-Multinomial team allocation (small, fast PyMC run)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")

from ffmodel.features import crossseason as cs
from ffmodel.models import volume_alloc as va

FIT_KW = dict(draws=150, tune=150, chains=2)


@pytest.fixture(scope="module")
def fitted():
    groups = cs.build_team_groups([2016, 2017, 2018, 2019, 2020], resource="target",
                                  source="legacy")
    return va.DirichletAllocation().fit(groups, **FIT_KW), groups


def test_predicted_shares_sum_to_one_per_group(fitted):
    model, groups = fitted
    shares = model.predict_shares(groups).mean(axis=1)
    totals = pd.Series(shares).groupby(groups["group_id"].to_numpy()).sum()
    assert np.allclose(totals.to_numpy(), 1.0, atol=1e-6)


def _group(players):
    rows = []
    for name, pos, hist in players:
        rows.append({
            "group_id": "G", "player_name": name, "position": pos,
            "hist_share": hist, "prior_share": hist, "late_share": hist,
            "is_rookie": 0, "team_change": 0, "age": 26.0,
            "draft_value": 0.0, "years_since_draft": 3.0,
            "contract_value": 0.0, "contract_year": 1.0,
        })
    return pd.DataFrame(rows)


def test_adding_a_claimant_dilutes_the_others(fitted):
    model, _ = fitted
    base = _group([("A", "WR", 0.25), ("B", "WR", 0.15), ("C", "TE", 0.10)])
    more = _group([("A", "WR", 0.25), ("B", "WR", 0.15), ("C", "TE", 0.10),
                   ("D", "WR", 0.20)])
    s_base = model.predict_shares(base).mean(axis=1)
    s_more = model.predict_shares(more).mean(axis=1)
    # Each original player's share must fall once a new claimant joins the pie.
    assert (s_more[:3] < s_base).all()
    assert np.isclose(s_more.sum(), 1.0)


def test_more_history_gets_more_share(fitted):
    model, _ = fitted
    g = _group([("Star", "WR", 0.30), ("Depth", "WR", 0.05)])
    s = model.predict_shares(g).mean(axis=1)
    assert s[0] > s[1]
