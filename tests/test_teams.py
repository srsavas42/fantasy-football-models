"""Canonical team codes and relocation collapse."""

import pandas as pd

from ffmodel.data.schema import conform
from ffmodel.data.teams import normalize_team


def test_style_aliases_map_to_canonical():
    assert normalize_team("GNB") == "GB"
    assert normalize_team("KAN") == "KC"
    assert normalize_team("SFO") == "SF"
    assert normalize_team("NWE") == "NE"
    assert normalize_team("TAM") == "TB"


def test_relocations_collapse_to_one_franchise():
    assert normalize_team("STL") == normalize_team("LAR") == "LAR"
    assert normalize_team("SD") == normalize_team("SDG") == normalize_team("LAC") == "LAC"
    assert normalize_team("OAK") == normalize_team("LVR") == "LV"
    assert normalize_team("WFT") == "WAS"


def test_unknown_and_missing_pass_through():
    assert normalize_team("XYZ") == "XYZ"
    assert normalize_team(None) is None


def test_conform_normalizes_team_column():
    df = pd.DataFrame({
        "player_name": ["A", "B"], "position": ["WR", "WR"], "team": ["STL", "GNB"],
        "season": [2015, 2015], "week": [1, 1],
    })
    out = conform(df)
    assert out["team"].tolist() == ["LAR", "GB"]
