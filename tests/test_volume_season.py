"""Cross-season Beta model + projection smoke tests (small, fast PyMC runs)."""

import numpy as np
import pytest

pytest.importorskip("pymc")

from ffmodel.features import crossseason as cs
from ffmodel.models import volume_season as vs
from ffmodel.projections import season_volume as sv

# Fit once for the module; tiny sampler settings keep the suite quick.
FIT_KW = dict(draws=150, tune=150, chains=2)


@pytest.fixture(scope="module")
def transitions():
    return cs.build_transitions([2017, 2018, 2019, 2020], source="legacy")


@pytest.fixture(scope="module")
def target_model(transitions):
    return vs.fit_target_share(transitions, **FIT_KW)


def test_predict_samples_shape_and_range(transitions, target_model):
    head = transitions[transitions["position"].isin(target_model.positions)].head(20)
    s = target_model.predict_samples(head)
    assert s.shape[0] == 20
    assert (s > 0).all() and (s < 1).all()  # Beta support


def test_predict_quantiles_are_ordered(transitions, target_model):
    skill = transitions[transitions["position"].isin(target_model.positions)]
    q = target_model.predict_quantiles(skill.head(30))
    assert (q["p10"] <= q["p50"] + 1e-9).all()
    assert (q["p50"] <= q["p90"] + 1e-9).all()


def test_breakout_report_probabilities_valid(transitions, target_model):
    rep = sv.breakout_report(transitions, target_model, threshold=0.05)
    assert rep["p_breakout"].between(0, 1).all()
    assert rep["p_decline"].between(0, 1).all()
    # Sorted by breakout probability, descending.
    assert rep["p_breakout"].is_monotonic_decreasing


def test_project_next_season_bands_ordered(transitions, target_model):
    proj = sv.project_next_season(transitions.head(30), target_model)
    assert (proj["proj_opp_p10"] <= proj["proj_opp_p50"] + 1e-9).all()
    assert (proj["proj_opp_p50"] <= proj["proj_opp_p90"] + 1e-9).all()
    assert (proj["proj_opp_mean"] >= 0).all()


def test_pass_stream_projects_qbs_separately(transitions):
    # Passes are their own stream: QBs are absent from the target model but
    # projectable via fit_pass_share, with shares in the Beta support.
    assert "next_pass_share" in transitions.columns
    qb = transitions[transitions["position"] == "QB"]
    assert len(qb) > 0
    model = vs.fit_pass_share(transitions, **FIT_KW)
    s = model.predict_samples(qb.head(10))
    assert s.shape[0] == 10
    # QB pass_share is bimodal (starter vs backup); a near-degenerate Beta can
    # sample exactly 0/1, so shares are valid in the closed [0, 1].
    assert (s >= 0).all() and (s <= 1).all()


def test_projections_exclude_qbs(transitions, target_model):
    # Opportunity projections/breakout must not include QBs.
    rep = sv.breakout_report(transitions, target_model)
    assert not (rep["position"] == "QB").any()


def test_more_competition_lowers_carry_projection(transitions):
    # The carry model learns a strongly negative competition coefficient, so
    # adding incoming RB competition must reduce projected carry share.
    rbs = transitions[transitions["position"] == "RB"]
    model = vs.fit_carry_share(transitions, **FIT_KW)
    low = rbs.copy();  low["incoming_comp_carry"] = 0.0
    high = rbs.copy(); high["incoming_comp_carry"] = 0.5
    mean_low = model.predict_samples(low).mean()
    mean_high = model.predict_samples(high).mean()
    assert mean_high < mean_low
