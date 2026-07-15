"""How sticky is volume, and what moves it year to year?

Offline analysis on the legacy CSVs, informing the cross-season model's spec:
  1. Stickiness: correlation of prior vs next share for stayers vs team-changers.
  2. Age: mean share and mean next-year *change* in share by age bucket.
  3. Experience: mean next-year change in share by years in the league.
  4. QB tendency: how sticky a team's positional target distribution is, and
     whether it is a portable QB trait or a team/scheme effect.
  5. History: does a multi-year sequence beat single-year, and does momentum
     mean-revert?

Run: python scripts/analyze_usage_dynamics.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ffmodel.data import load_player_weeks
from ffmodel.features import crossseason as cs

SEASONS = range(2010, 2021)
QB_SEASONS = range(2008, 2022)


def stickiness(t: pd.DataFrame) -> None:
    print("1) Stickiness — corr(prior share, next share), stayers vs team-changers:")
    for pos in ("WR", "RB", "TE"):
        p = t[t["position"] == pos]
        resources = [("target", "target_share", "next_target_share")]
        if pos == "RB":
            resources.append(("carry", "carry_share", "next_carry_share"))
        for res, pc, nc in resources:
            stay, move = p[p["team_change"] == 0], p[p["team_change"] == 1]
            print(f"   {pos} {res:6s}: stayers r={stay[pc].corr(stay[nc]):.3f} "
                  f"(n={len(stay)})   movers r={move[pc].corr(move[nc]):.3f} (n={len(move)})")
    print("   -> volume is meaningfully less predictable after a team change.\n")


def age_curve(t: pd.DataFrame) -> None:
    print("2) Age — mean opportunity share and mean Δ(share) next year, by age:")
    t = t.copy()
    t["age"] = pd.to_numeric(t["age"], errors="coerce")
    t["opp"] = t["target_share"] + t["carry_share"]
    t["dopp"] = (t["next_target_share"] + t["next_carry_share"]) - t["opp"]
    t["agebin"] = pd.cut(t["age"], [20, 23, 25, 27, 29, 32, 40])
    for pos in ("WR", "RB", "TE"):
        g = t[t["position"] == pos].groupby("agebin").agg(
            share=("opp", "mean"), delta=("dopp", "mean"), n=("opp", "size")
        )
        print(f"   {pos}:")
        print(g.round(3).to_string().replace("\n", "\n   "))
    print("   -> decline accelerates with age; RBs erode earliest and hardest.\n")


def experience_curve(t: pd.DataFrame) -> None:
    print("3) Experience — mean Δ(opportunity share) next year, by years in league:")
    t = t.copy()
    t["opp"] = t["target_share"] + t["carry_share"]
    t["dopp"] = (t["next_target_share"] + t["next_carry_share"]) - t["opp"]
    t["exp"] = pd.to_numeric(t["experience"], errors="coerce").clip(0, 8)
    for pos in ("WR", "RB"):
        g = t[t["position"] == pos].groupby("exp").agg(
            delta=("dopp", "mean"), n=("opp", "size")
        )
        print(f"   {pos}: " + "  ".join(f"exp{int(i)}:{r.delta:+.3f}" for i, r in g.iterrows()))
    print()


def _team_positional_targets(pw: pd.DataFrame) -> pd.DataFrame:
    """Per team-season: fraction of team targets to each of WR/RB/TE, plus the
    primary QB (most attempts)."""
    tot = pw.groupby(["season", "team"]).agg(tot=("targets", "sum")).reset_index()
    pt = (
        pw[pw["position"].isin(["WR", "RB", "TE"])]
        .groupby(["season", "team", "position"])["targets"].sum().reset_index()
        .merge(tot, on=["season", "team"])
    )
    pt["frac"] = pt["targets"] / pt["tot"]
    wide = pt.pivot_table(index=["season", "team"], columns="position",
                          values="frac").reset_index().fillna(0.0)
    qb = (
        pw[pw["position"] == "QB"]
        .groupby(["season", "team"])
        .apply(lambda g: g.loc[g["pass_att"].idxmax(), "player_name"]
               if g["pass_att"].max() > 20 else None)
        .reset_index(name="qb").dropna()
    )
    return wide.merge(qb, on=["season", "team"])


def qb_tendency(pw: pd.DataFrame) -> None:
    print("4) QB tendency — stickiness of a team's positional target distribution:")
    wide = _team_positional_targets(pw)
    nxt = wide.copy(); nxt["season"] = nxt["season"] - 1
    m = wide.merge(nxt, on=["season", "team"], suffixes=("", "_next"))
    m["qb_stayed"] = (m["qb"] == m["qb_next"]).astype(int)
    for p in ("TE", "WR", "RB"):
        st, ch = m[m["qb_stayed"] == 1], m[m["qb_stayed"] == 0]
        print(f"   {p}: overall r={m[p].corr(m[p + '_next']):.3f}  |  "
              f"same QB r={st[p].corr(st[p + '_next']):.3f}  "
              f"QB changed r={ch[p].corr(ch[p + '_next']):.3f}")
    # Does the tendency travel with a QB who switches teams?
    w2 = wide.copy(); w2["season"] = w2["season"] - 1
    j = wide.merge(w2, on="qb", suffixes=("", "_n"))
    j = j[j["season_n"] == j["season"] + 1]
    moved = j[j["team"] != j["team_n"]]
    travel = "  ".join(f"{p} r={moved[p].corr(moved[p + '_n']):.3f}" for p in ("TE", "WR", "RB"))
    print(f"   Travels with QB across a team change (n={len(moved)}): {travel}")
    print("   -> distribution is sticky at the TEAM level (scheme + personnel);")
    print("      only weakly a portable QB trait.\n")


def history_value(usage: pd.DataFrame) -> None:
    print("5) History — is a multi-year sequence better than last year alone?")
    u = usage[usage["position"].isin(["WR", "RB", "TE"])].sort_values(["key", "season"]).copy()
    u["opp"] = u["target_share"] + u["carry_share"]
    u["last1"] = u.groupby("key")["opp"].shift(1)
    u["ewma3"] = u.groupby("key")["opp"].transform(
        lambda s: s.shift(1).ewm(span=3, min_periods=1).mean()
    )
    u["trend"] = u.groupby("key")["opp"].transform(lambda s: s.shift(1) - s.shift(2))
    sub = u.dropna(subset=["last1", "ewma3", "opp"])
    print(f"   corr(last-year share, actual) = {sub['last1'].corr(sub['opp']):.3f}")
    print(f"   corr(3yr-EWMA share, actual)  = {sub['ewma3'].corr(sub['opp']):.3f}")
    s2 = sub.dropna(subset=["trend"])
    r = np.corrcoef(s2["trend"], s2["opp"] - s2["last1"])[0, 1]
    print(f"   corr(prior trend, residual change) = {r:.3f}  "
          "(<0 => momentum mean-reverts)\n")


def main() -> None:
    t = cs.build_transitions(SEASONS, source="legacy")
    print(f"Transitions: {len(t):,} ({t['transition'].nunique()} year-pairs)\n")
    stickiness(t)
    age_curve(t)
    experience_curve(t)
    qb_tendency(load_player_weeks(QB_SEASONS, source="legacy"))
    history_value(cs.season_usage(SEASONS, source="legacy"))


if __name__ == "__main__":
    main()
