"""Contract features (synthetic fixtures — the OTC feed is data-gated)."""

import numpy as np
import pandas as pd

from ffmodel.features import contracts as ct
from ffmodel.features import crossseason as cs
from ffmodel.features.investment import add_investment


def _contracts():
    return pd.DataFrame([
        {"player": "Star WR", "position": "WR", "team": "A", "year_signed": 2018,
         "years": 4, "value": 80.0, "guaranteed": 60.0, "apy_cap_pct": 0.12},
        {"player": "Cheap WR", "position": "WR", "team": "A", "year_signed": 2019,
         "years": 4, "value": 8.0, "guaranteed": 7.0, "apy_cap_pct": 0.02},
        # A bigger second deal for Star WR that must supersede the first from 2022.
        {"player": "Star WR", "position": "WR", "team": "B", "year_signed": 2022,
         "years": 3, "value": 90.0, "guaranteed": 70.0, "apy_cap_pct": 0.15},
    ])


def test_contract_covers_signed_years_with_timeline():
    f = ct.season_contract_features(_contracts(), range(2018, 2022))
    star = f[f["player_name"] == "Star WR"].set_index("season")
    assert set(star.index) == {2018, 2019, 2020, 2021}
    assert star.loc[2018, "contract_year"] == 0
    assert star.loc[2021, "contract_year"] == 3
    assert star.loc[2018, "years_remaining"] == 4
    assert star.loc[2021, "years_remaining"] == 1


def test_most_recent_deal_supersedes():
    f = ct.season_contract_features(_contracts(), range(2018, 2025))
    star = f[f["player_name"] == "Star WR"].set_index("season")
    # 2022 is covered by the new deal, not the expired 2018 one.
    assert star.loc[2022, "contract_year"] == 0
    assert star.loc[2022, "years_remaining"] == 3


def test_contract_value_standardized_within_position():
    f = ct.season_contract_features(_contracts(), range(2018, 2022))
    # The expensive deal is above the position mean, the cheap one below.
    star = f[(f["player_name"] == "Star WR") & (f["season"] == 2019)]["contract_value"].iloc[0]
    cheap = f[(f["player_name"] == "Cheap WR") & (f["season"] == 2019)]["contract_value"].iloc[0]
    assert star > 0 > cheap


def test_guaranteed_pct_bounded():
    f = ct.season_contract_features(_contracts(), range(2018, 2025))
    assert f["guaranteed_pct"].between(0, 1).all()


def test_add_investment_attaches_contracts_when_provided():
    df = pd.DataFrame({
        "player_name": ["Star WR", "Cheap WR"],
        "position": ["WR", "WR"],
        "season": [2019, 2019],
    })
    out = add_investment(df, source="legacy", contracts=_contracts())
    assert set(ct.CONTRACT_FEATURES) <= set(out.columns)
    assert out.loc[out["player_name"] == "Star WR", "contract_value"].iloc[0] > 0


def test_add_investment_zero_contracts_offline():
    # source="legacy" and no contracts -> features present and zero, no crash.
    df = pd.DataFrame({"player_name": ["X"], "position": ["WR"], "season": [2019]})
    out = add_investment(df, source="legacy")
    for col in ct.CONTRACT_FEATURES:
        assert (out[col] == 0).all()
