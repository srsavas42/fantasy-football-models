"""Cross-season feature correctness: vacated share, returning-only, no leakage."""

import numpy as np
import pandas as pd

from ffmodel.features import crossseason as cs


def _usage(rows):
    df = pd.DataFrame(
        rows, columns=["player_name", "position", "season", "team", "target_share", "carry_share"]
    )
    df["key"] = cs.player_key(df)
    return df


def test_vacated_equals_departed_share_sum():
    # Team T in 2019: A/B/C. In 2020 only A returns -> B+C vacate their shares.
    u = _usage([
        ["A", "WR", 2019, "T", 0.40, 0.0],
        ["B", "WR", 2019, "T", 0.30, 0.0],
        ["C", "RB", 2019, "T", 0.10, 0.50],
        ["A", "WR", 2020, "T", 0.45, 0.0],
    ])
    vac = cs.vacated_opportunity(u, 2019)
    row = vac[vac["team"] == "T"].iloc[0]
    assert np.isclose(row["vacated_target_share"], 0.30 + 0.10)  # B + C targets
    assert np.isclose(row["vacated_carry_share"], 0.50)          # C carries
    assert row["next_season"] == 2020


def test_returning_player_does_not_vacate():
    u = _usage([
        ["A", "WR", 2019, "T", 0.5, 0.0],
        ["A", "WR", 2020, "T", 0.5, 0.0],
    ])
    vac = cs.vacated_opportunity(u, 2019)
    assert vac.empty or np.isclose(vac.iloc[0]["vacated_target_share"], 0.0)


def test_transitions_are_returning_only():
    t = cs.build_transitions([2018, 2019, 2020], source="legacy")
    usage = cs.season_usage([2018, 2019, 2020], source="legacy")
    for _, r in t.sample(min(40, len(t)), random_state=0).iterrows():
        y, yp1 = map(int, r["transition"].split("->"))
        in_y = ((usage["key"] == r["key"]) & (usage["season"] == y)).any()
        in_yp1 = ((usage["key"] == r["key"]) & (usage["season"] == yp1)).any()
        assert in_y and in_yp1


def test_predictors_from_year_y_labels_from_yp1():
    # A transition's prior share must equal the player's season-Y usage, and its
    # label must equal season-(Y+1) usage: predictors never read the future.
    usage = cs.season_usage([2018, 2019], source="legacy")
    t = cs.build_transitions([2018, 2019], source="legacy")
    r = t.iloc[0]
    y, yp1 = map(int, r["transition"].split("->"))
    uy = usage[(usage["key"] == r["key"]) & (usage["season"] == y)].iloc[0]
    uyp1 = usage[(usage["key"] == r["key"]) & (usage["season"] == yp1)].iloc[0]
    assert np.isclose(r["target_share"], uy["target_share"])
    assert np.isclose(r["next_target_share"], uyp1["target_share"])


def test_career_history_is_causal_ewma():
    # hist at each season must use only that season and earlier (no future leak),
    # and match a manual EWMA of the sequence so far.
    u = pd.DataFrame({
        "player_name": ["P"] * 4,
        "position": ["WR"] * 4,
        "season": [2016, 2017, 2018, 2019],
        "team": ["A"] * 4,
        "target_share": [0.10, 0.20, 0.30, 0.40],
        "carry_share": [0.0] * 4,
    })
    u["key"] = cs.player_key(u)
    out = cs.add_career_history(u).sort_values("season")
    expected = pd.Series([0.10, 0.20, 0.30, 0.40]).ewm(span=3, min_periods=1).mean()
    assert np.allclose(out["hist_target_share"].to_numpy(), expected.to_numpy())
    # A rising sequence has positive trend; first season's trend is 0 (no prior).
    assert out["target_trend"].iloc[0] == 0.0
    assert (out["target_trend"].iloc[1:] > 0).all()


def test_team_groups_shares_sum_to_one():
    g = cs.build_team_groups([2017, 2018, 2019], resource="target", source="legacy")
    assert not g.empty
    sums = g.groupby("group_id")["label_share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)
    # Rookies appear as group members with zero usage history.
    rookies = g[g["is_rookie"] == 1]
    assert len(rookies) > 0
    assert (rookies["hist_share"] == 0).all()


def test_team_groups_carry_resource():
    g = cs.build_team_groups([2018, 2019], resource="carry", source="legacy")
    sums = g.groupby("group_id")["label_share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)
    # Carry groups are backfield-heavy but include QBs/WRs who run.
    assert set(g["position"].unique()) <= {"RB", "QB", "WR"}


def test_shares_within_unit_interval():
    t = cs.build_transitions([2018, 2019, 2020], source="legacy")
    for col in ("target_share", "carry_share", "next_target_share", "next_carry_share"):
        assert (t[col] >= -1e-9).all() and (t[col] <= 1 + 1e-9).all()
