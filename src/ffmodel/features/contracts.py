"""Contract-based team investment for veterans.

Draft capital fades within a few years (`features/investment.py`), so a veteran's
organizational commitment lives in their *contract*: a big guaranteed second deal
is a team betting on volume. This turns the OverTheCap contract table
(`data.ingest.load_contracts`, one row per contract) into per (player, season)
signals the share model can consume:

  contract_value    APY as % of the salary cap, standardized WITHIN position so a
                    QB and an RB deal are comparable (cap% is already era-adjusted)
  guaranteed_pct    fraction of the contract guaranteed — commitment intensity
  contract_year     seasons into the deal (0 = signing year)
  years_remaining   seasons left on the deal

As with draft capital, the model is fed raw signals + a timeline interaction and
left to learn how commitment fades over a deal, rather than a hard-coded decay.
This module is data-gated (OTC is github-hosted, blocked in this sandbox); it is
exercised here with synthetic fixtures and against real data locally.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CONTRACT_FEATURES = ["contract_value", "guaranteed_pct", "contract_year", "years_remaining"]


def _canon(contracts: pd.DataFrame) -> pd.DataFrame:
    df = contracts.rename(columns={"player": "player_name"}).copy()
    for c in ("year_signed", "years", "value", "apy", "guaranteed", "apy_cap_pct"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def season_contract_features(contracts: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """Per (player_name, position, season) contract signals for `seasons`.

    Expands each contract across the seasons it covers, keeps the most recently
    signed deal in force for each player-season, and standardizes contract size
    within position.
    """
    seasons = set(int(s) for s in seasons)
    df = _canon(contracts)
    df = df.dropna(subset=["year_signed", "years"])
    df = df[df["years"] >= 1]
    if df.empty:
        return pd.DataFrame(columns=["player_name", "position", "season", *CONTRACT_FEATURES])

    # Expand one row per (contract, covered season).
    n = df["years"].astype(int).clip(lower=1)
    exp = df.loc[df.index.repeat(n)].copy()
    exp["offset"] = exp.groupby(level=0).cumcount()
    exp["season"] = exp["year_signed"].astype(int) + exp["offset"]
    exp = exp[exp["season"].isin(seasons)]
    if exp.empty:
        return pd.DataFrame(columns=["player_name", "position", "season", *CONTRACT_FEATURES])

    # Most recently signed deal wins a given player-season.
    exp = exp.sort_values("year_signed").drop_duplicates(
        ["player_name", "position", "season"], keep="last"
    )
    exp["contract_year"] = exp["season"] - exp["year_signed"].astype(int)
    exp["years_remaining"] = exp["year_signed"].astype(int) + exp["years"].astype(int) - exp["season"]
    exp["guaranteed_pct"] = _safe_ratio(exp.get("guaranteed"), exp.get("value"))

    # Standardize APY-cap-% within position (comparable QB vs RB).
    exp["apy_cap_pct"] = pd.to_numeric(exp.get("apy_cap_pct"), errors="coerce")
    grp = exp.groupby("position")["apy_cap_pct"]
    mean, std = grp.transform("mean"), grp.transform("std").replace(0, np.nan)
    exp["contract_value"] = ((exp["apy_cap_pct"] - mean) / std).fillna(0.0)

    return exp[["player_name", "position", "season", *CONTRACT_FEATURES]].reset_index(drop=True)


def _safe_ratio(num, den) -> np.ndarray:
    if num is None or den is None:
        return np.zeros(0)
    num = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(den, errors="coerce").to_numpy(dtype=float)
    out = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))
    return np.clip(out, 0.0, 1.0)
