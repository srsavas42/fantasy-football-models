"""Cross-season (year-over-year) volume features for returning players.

Answers the pre-season question: which returning players will see a change in
season-level volume next year? The dominant driver is **vacated opportunity** —
the targets/carries freed when a teammate departs — computed here purely from
who appears on each team's roster across consecutive seasons, so it works fully
offline from the committed weekly/yearly CSVs.

Everything a transition row predicts *from* comes from season Y; the label comes
from season Y+1. No Y+1 information leaks into the predictors (the one exception,
by design, is vacated opportunity, which needs to know who left the team for
Y+1 — realized for backtests, from the known offseason roster when projecting).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.data import load_player_weeks
from ffmodel.data import legacy
from ffmodel.features.volume import SKILL_POSITIONS

LATE_SEASON_START_WEEK = 10  # weeks >= this define "late season" role signal

# Who competes for each resource's pie. Targets are split across all pass
# catchers; carries across everyone who runs (QBs scramble, WRs take sweeps).
TARGET_GROUP = ("WR", "TE", "RB")
CARRY_GROUP = ("RB", "QB", "WR")


def player_key(df: pd.DataFrame) -> pd.Series:
    """Stable cross-season player identity.

    Prefers the nflverse `player_id` (gsis) when present, which disambiguates
    players who share a name across seasons; falls back to name+position for the
    legacy CSVs, which carry no id.
    """
    name_pos = df["player_name"].astype(str) + "|" + df["position"].astype(str)
    if "player_id" in df.columns and df["player_id"].notna().any():
        pid = df["player_id"]
        return pid.where(pid.notna(), name_pos).astype(str)
    return name_pos


def _shares(players: pd.DataFrame, team_targets: float, team_carries: float) -> pd.DataFrame:
    players = players.copy()
    players["target_share"] = _safe_div(players["targets"], team_targets)
    players["carry_share"] = _safe_div(players["rush_att"], team_carries)
    return players


def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = float(den) if np.isscalar(den) else np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float),
                     where=np.asarray(den) != 0)


def season_usage(seasons: Iterable[int], source: str = "auto") -> pd.DataFrame:
    """Per (player, position, season) season-level usage on the player's main team.

    A player traded mid-year is attributed to the team where they saw the most
    opportunity; shares are relative to that team's full-season totals. Includes
    full-season and late-season (weeks >= 10) target/carry shares, games played,
    and age (from the yearly CSVs when available).
    """
    seasons = sorted(set(seasons))
    pw = load_player_weeks(seasons, source=source)
    pw["opportunity"] = pw["targets"] + pw["rush_att"]
    pw["is_active"] = (pw["pass_att"] + pw["opportunity"] + pw["receptions"] > 0).astype(int)

    # Attribute each (player, season) to the team with the most opportunity.
    by_pt = (
        pw.groupby(["player_name", "position", "season", "team"], dropna=False)["opportunity"]
        .sum()
        .reset_index()
    )
    main_team = (
        by_pt.sort_values("opportunity")
        .groupby(["player_name", "position", "season"], dropna=False)
        .tail(1)[["player_name", "position", "season", "team"]]
        .rename(columns={"team": "main_team"})
    )
    pw = pw.merge(main_team, on=["player_name", "position", "season"], how="left")

    rows = []
    for season in seasons:
        sea = pw[pw["season"] == season]
        team_tot = sea.groupby("team").agg(
            team_targets=("targets", "sum"), team_carries=("rush_att", "sum"),
            team_pass=("pass_att", "sum"),
        )
        late = sea[sea["week"] >= LATE_SEASON_START_WEEK]
        late_tot = late.groupby("team").agg(
            team_targets=("targets", "sum"), team_carries=("rush_att", "sum")
        )
        # One row per player on their main team.
        agg = (
            sea[sea["team"] == sea["main_team"]]
            .groupby(["player_name", "position", "season", "team"], dropna=False)
            .agg(targets=("targets", "sum"), rush_att=("rush_att", "sum"),
                 pass_att=("pass_att", "sum"), games=("is_active", "sum"))
            .reset_index()
        )
        agg["target_share"] = [
            _safe_div(t, team_tot.loc[tm, "team_targets"]) if tm in team_tot.index else 0.0
            for t, tm in zip(agg["targets"], agg["team"])
        ]
        agg["carry_share"] = [
            _safe_div(c, team_tot.loc[tm, "team_carries"]) if tm in team_tot.index else 0.0
            for c, tm in zip(agg["rush_att"], agg["team"])
        ]
        # Pass share = fraction of team pass attempts the player threw (QBs).
        agg["pass_share"] = [
            _safe_div(p, team_tot.loc[tm, "team_pass"]) if tm in team_tot.index else 0.0
            for p, tm in zip(agg["pass_att"], agg["team"])
        ]
        # Late-season shares (role signal that projects forward).
        late_agg = (
            late[late["team"] == late["main_team"]]
            .groupby(["player_name", "position"], dropna=False)
            .agg(late_targets=("targets", "sum"), late_rush=("rush_att", "sum"),
                 late_team=("team", "first"))
            .reset_index()
        )
        late_agg["late_target_share"] = [
            _safe_div(t, late_tot.loc[tm, "team_targets"]) if tm in late_tot.index else 0.0
            for t, tm in zip(late_agg["late_targets"], late_agg["late_team"])
        ]
        late_agg["late_carry_share"] = [
            _safe_div(c, late_tot.loc[tm, "team_carries"]) if tm in late_tot.index else 0.0
            for c, tm in zip(late_agg["late_rush"], late_agg["late_team"])
        ]
        agg = agg.merge(
            late_agg[["player_name", "position", "late_target_share", "late_carry_share"]],
            on=["player_name", "position"], how="left",
        )
        rows.append(agg)

    usage = pd.concat(rows, ignore_index=True)
    usage[["late_target_share", "late_carry_share"]] = usage[
        ["late_target_share", "late_carry_share"]
    ].fillna(0.0)
    # The list-comprehension assignments above yield object dtype; coerce the
    # share columns to float so downstream arithmetic stays numeric.
    for col in ("target_share", "carry_share", "pass_share",
                "late_target_share", "late_carry_share"):
        usage[col] = usage[col].astype(float)
    # Team-relative opportunity share (targets + carries, each vs team totals).
    usage["opportunity_share"] = usage["target_share"] + usage["carry_share"]
    usage["key"] = player_key(usage)
    usage = _merge_age(usage, seasons)
    return usage


def _merge_age(usage: pd.DataFrame, seasons) -> pd.DataFrame:
    """Attach age from the yearly CSVs; approximate experience within the window."""
    yearly = legacy.load_yearly(seasons)
    if "age" not in [c.lower() for c in yearly.columns]:
        # legacy.load_yearly conforms to the stat schema and drops Age; read raw.
        age = _raw_yearly_age(seasons)
    else:
        age = yearly.rename(columns={"Age": "age"})[["player_name", "season", "age"]]
    usage = usage.merge(age, on=["player_name", "season"], how="left")
    # Experience proxy: seasons since first appearance in the loaded window.
    first = usage.groupby("key")["season"].transform("min")
    usage["experience"] = usage["season"] - first
    return usage


def _raw_yearly_age(seasons) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = legacy.LEGACY_YEARLY_DIR / f"{season}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw = raw.loc[:, ~raw.columns.str.match(r"^Unnamed")]
        if "Age" not in raw.columns:
            continue
        sub = raw[["Player", "Age"]].rename(columns={"Player": "player_name", "Age": "age"})
        sub["season"] = season
        frames.append(sub.drop_duplicates("player_name"))
    if not frames:
        return pd.DataFrame(columns=["player_name", "season", "age"])
    return pd.concat(frames, ignore_index=True)


def vacated_opportunity(usage: pd.DataFrame, from_season: int) -> pd.DataFrame:
    """Share vacated on each team entering `from_season + 1`.

    For each team, sums the season-`from_season` target/carry shares of players
    who are NOT on that team in `from_season + 1` (departed). Keyed by team with
    `next_season = from_season + 1`.
    """
    y, yp1 = from_season, from_season + 1
    cur = usage[usage["season"] == y]
    nxt = usage[usage["season"] == yp1]
    present_next = set(zip(nxt["key"], nxt["team"]))

    departed = cur[~cur.apply(lambda r: (r["key"], r["team"]) in present_next, axis=1)]
    vac = departed.groupby("team").agg(
        vacated_target_share=("target_share", "sum"),
        vacated_carry_share=("carry_share", "sum"),
    ).reset_index()
    vac["next_season"] = yp1
    return vac


def add_career_history(usage: pd.DataFrame) -> pd.DataFrame:
    """Add multi-year form and momentum to each (player, season) row.

    Rather than describe a player by their single most recent season, summarize
    the whole sequence so far:
      * `hist_target_share` / `hist_carry_share` — EWMA (span 3) of share over
        the player's seasons up to and including the current one.
      * `target_trend` / `carry_trend` — one-year change (this season minus last).
        Analysis shows momentum mean-reverts, so a positive trend predicts a
        pullback next year; the model uses the sign accordingly.

    All quantities use only seasons <= the row's season, so nothing leaks from
    the transition's target year (Y+1).
    """
    u = usage.sort_values(["key", "season"], kind="stable").copy()
    g = u.groupby("key", sort=False)
    trend_cols = []
    for res in ("target_share", "carry_share", "pass_share"):
        if res not in u.columns:
            continue
        stem = res.replace("_share", "")
        u[f"hist_{stem}_share"] = g[res].transform(
            lambda s: s.ewm(span=3, min_periods=1).mean()
        )
        u[f"{stem}_trend"] = g[res].transform(lambda s: s - s.shift(1))
        trend_cols.append(f"{stem}_trend")
    u[trend_cols] = u[trend_cols].fillna(0.0)
    return u


def _safe_draft_capital(seasons, source):
    """Load draft capital, degrading to None if the source is unavailable."""
    from ffmodel.features.draft import load_draft_capital

    try:
        dc = load_draft_capital(seasons, source=source)
        return dc if not dc.empty else None
    except Exception:
        return None


def incoming_competition(
    usage: pd.DataFrame, from_season: int, draft_capital: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Competition *arriving* on each team entering `from_season + 1`.

    Two sources of new mouths to feed, both offsetting vacated opportunity:
      * incoming veterans — players on team T in Y+1 who were elsewhere in Y;
        their Y share (on the old team) proxies the claim they bring.
      * incoming rookies — drafted to T for Y+1; claim proxied by draft capital
        (`features.draft.expected_rookie_claim`).

    Returns per team: `incoming_comp_target` / `incoming_comp_carry`, keyed with
    `next_season = from_season + 1`.
    """
    from ffmodel.features.draft import expected_rookie_claim

    y, yp1 = from_season, from_season + 1
    cur = usage[usage["season"] == y]
    nxt = usage[usage["season"] == yp1]
    prev_membership = set(zip(cur["key"], cur["team"]))
    prior_share = cur.set_index("key")[["target_share", "carry_share"]]

    # Incoming veterans: on T in Y+1, not on T in Y, but with Y usage elsewhere.
    is_new = np.array(
        [(k, t) not in prev_membership for k, t in zip(nxt["key"], nxt["team"])]
    )
    incoming = nxt[is_new]
    incoming = incoming[incoming["key"].isin(prior_share.index)]
    vet = incoming.merge(
        prior_share.rename(columns={"target_share": "vt", "carry_share": "vc"}),
        left_on="key", right_index=True, how="left",
    )
    # Competition is split by resource via each player's own share footprint:
    # everyone who catches passes adds target competition, everyone who runs adds
    # carry competition. This already separates e.g. a WR (mostly targets) from a
    # RB (mostly carries) without excluding any position from the run game, since
    # QBs (scrambles) and WRs (jet sweeps) carry the ball too.
    vet_agg = vet.groupby("team").agg(
        incoming_comp_target=("vt", "sum"), incoming_comp_carry=("vc", "sum")
    )

    # Incoming rookies: drafted to T for the Y+1 season.
    rook_agg = pd.DataFrame(columns=["incoming_comp_target", "incoming_comp_carry"])
    if draft_capital is not None and not draft_capital.empty:
        rk = draft_capital[draft_capital["season"] == yp1].copy()
        if not rk.empty:
            claims = rk.apply(
                lambda r: expected_rookie_claim(r["overall_pick"], r["position"]),
                axis=1, result_type="expand",
            )
            rk["incoming_comp_target"] = claims[0].to_numpy()
            rk["incoming_comp_carry"] = claims[1].to_numpy()
            rook_agg = rk.groupby("team")[
                ["incoming_comp_target", "incoming_comp_carry"]
            ].sum()

    comp = vet_agg.add(rook_agg, fill_value=0.0).reset_index().rename(
        columns={"index": "team"}
    )
    comp["next_season"] = yp1
    return comp


def enrich_usage(seasons: Iterable[int], source: str = "auto") -> pd.DataFrame:
    """Per (player, season) usage with multi-year history and investment attached.

    The shared feature table behind both the per-player transitions and the
    team-group allocation. Computed for every season so it can serve either as a
    label year (Y+1) or a feature year (Y).
    """
    from ffmodel.features.investment import add_investment

    usage = season_usage(seasons, source=source)
    usage = usage[usage["position"].isin(SKILL_POSITIONS + ("QB",))]
    usage = add_career_history(usage)
    usage = add_investment(usage, source=source)
    return usage


def build_team_groups(
    seasons: Iterable[int], resource: str = "target", source: str = "auto"
) -> pd.DataFrame:
    """Team-position roster groups for the Dirichlet allocation.

    One row per (team, Y+1, claimant) for each player who saw `resource` volume
    in Y+1 — the *group* over which next-season shares must sum to 1. Labels
    (`label_count`, `label_share`) are from Y+1; usage-history features
    (`hist_share`, `prior_share`, `late_share`) are strictly from Y (0 for
    rookies with no Y usage); investment and age are as-of the projection year
    Y+1 (known then, and defined for rookies). `group_id` identifies the simplex;
    vacated/competition are *not* columns here — they're structural (departed
    players are absent; adding claimants dilutes the softmax).
    """
    resource = resource.lower()
    # (group positions, count column, hist col, prior col, late col). Passes have
    # no late-season split, so they reuse the season level as the "late" signal.
    spec = {
        "target": (TARGET_GROUP, "targets", "hist_target_share", "target_share",
                   "late_target_share"),
        "carry": (CARRY_GROUP, "rush_att", "hist_carry_share", "carry_share",
                  "late_carry_share"),
        "pass": (("QB",), "pass_att", "hist_pass_share", "pass_share", "pass_share"),
    }
    if resource not in spec:
        raise ValueError(f"unknown resource {resource!r}; expected target/carry/pass")
    positions, count_col, hist_col, prior_col, late_col = spec[resource]

    usage = enrich_usage(seasons, source=source)
    seasons = sorted(set(seasons))
    keep_invest = ["age", "draft_value", "years_since_draft", "contract_value", "contract_year"]

    out = []
    for y in seasons[:-1]:
        yp1 = y + 1
        if yp1 not in seasons:
            continue
        grp = usage[
            (usage["season"] == yp1)
            & (usage["position"].isin(positions))
            & (usage[count_col] > 0)
        ].copy()
        if grp.empty:
            continue
        grp["team_total"] = grp.groupby("team")[count_col].transform("sum")
        grp["label_share"] = grp[count_col] / grp["team_total"]

        # Usage history strictly from Y (Y+1 usage is the label — no leakage).
        # `pass` reuses one column for prior + late, so build the rename carefully.
        ren = {hist_col: "hist_share", prior_col: "prior_share", "team": "team_y"}
        if late_col not in ren:
            ren[late_col] = "late_share"
        src = list(dict.fromkeys(["key", "team", hist_col, prior_col, late_col]))
        yf = usage[usage["season"] == y][src].rename(columns=ren)
        if "late_share" not in yf.columns:  # late aliased an already-renamed column
            yf["late_share"] = yf["prior_share"]
        m = grp.merge(yf, on="key", how="left")
        m["is_rookie"] = m["hist_share"].isna().astype(int)  # no Y usage at all
        for c in ("hist_share", "prior_share", "late_share"):
            m[c] = m[c].fillna(0.0)
        m["team_change"] = ((m["team_y"].notna()) & (m["team_y"] != m["team"])).astype(int)
        m["group_id"] = m["team"].astype(str) + "_" + str(yp1)
        m["next_season"] = yp1
        m["resource"] = resource
        m = m.rename(columns={count_col: "label_count"})
        cols = (["group_id", "key", "player_name", "position", "team", "next_season",
                 "label_count", "team_total", "label_share", "hist_share", "prior_share",
                 "late_share", "team_change", "is_rookie", "resource"] + keep_invest)
        out.append(m[[c for c in cols if c in m.columns]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_transitions(seasons: Iterable[int], source: str = "auto") -> pd.DataFrame:
    """One row per returning player transition (Y -> Y+1) with predictors + labels.

    Predictors are from season Y; labels (`next_*`) from Y+1. Only players who
    appear in both seasons (returning) are kept. Restricted to skill positions.
    """
    seasons = sorted(set(seasons))
    usage = season_usage(seasons, source=source)
    # QBs are kept so passes can be projected as their own stream; the per-stream
    # fit_* factories filter to the positions each model applies to.
    usage = usage[usage["position"].isin(SKILL_POSITIONS + ("QB",))]
    usage = add_career_history(usage)
    draft_capital = _safe_draft_capital(seasons, source)

    out = []
    for y in seasons[:-1]:
        yp1 = y + 1
        if yp1 not in seasons:
            continue
        cur = usage[usage["season"] == y].copy()
        nxt = usage[usage["season"] == yp1].copy()
        vac = vacated_opportunity(usage, y)
        comp = incoming_competition(usage, y, draft_capital)

        merged = cur.merge(
            nxt[["key", "team", "target_share", "carry_share", "pass_share",
                 "opportunity_share"]],
            on="key", suffixes=("", "_next"),
        )
        merged["team_change"] = (merged["team"] != merged["team_next"]).astype(int)
        # Vacated share and incoming competition both belong to the Y+1 team.
        merged = merged.merge(
            vac.rename(columns={"team": "team_next"})[
                ["team_next", "vacated_target_share", "vacated_carry_share"]
            ],
            on="team_next", how="left",
        )
        merged = merged.merge(
            comp.rename(columns={"team": "team_next"})[
                ["team_next", "incoming_comp_target", "incoming_comp_carry"]
            ],
            on="team_next", how="left",
        )
        merged["transition"] = f"{y}->{yp1}"
        out.append(merged)

    if not out:
        return pd.DataFrame()
    trans = pd.concat(out, ignore_index=True)
    for col in ("vacated_target_share", "vacated_carry_share",
                "incoming_comp_target", "incoming_comp_carry"):
        trans[col] = trans[col].fillna(0.0)
    # Net available opportunity: freed volume minus arriving competition.
    trans["net_target_opportunity"] = (
        trans["vacated_target_share"] - trans["incoming_comp_target"]
    )
    trans["net_carry_opportunity"] = (
        trans["vacated_carry_share"] - trans["incoming_comp_carry"]
    )
    trans = trans.rename(
        columns={
            "target_share_next": "next_target_share",
            "carry_share_next": "next_carry_share",
            "pass_share_next": "next_pass_share",
            "opportunity_share_next": "next_opportunity_share",
        }
    )
    # Team investment (draft capital, decayed) as-of season Y for each player.
    from ffmodel.features.investment import add_investment

    trans = add_investment(trans, source=source)
    return trans
