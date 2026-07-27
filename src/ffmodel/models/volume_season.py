"""Hierarchical Beta model for next-season opportunity share (returning players).

Predicts a returning player's season-Y+1 share (target share or carry share)
from season-Y signals. Shares live in [0, 1], so the likelihood is a Beta whose
mean is a logit-linear function of the predictors, with intercepts partially
pooled across positions (small-sample positions borrow strength). The posterior
predictive gives each player a full next-season share distribution, from which
projection quantiles and breakout probabilities are read off directly.

Persistence uses a level + shrinkage decomposition of the full sequence:
predicted logit-share ~= hist + w*(prior - hist), so `hist` is the multi-year
anchor and the `excess_vs_hist` weight w in (0,1) says how much a recent spike
sticks (w->1) vs reverts toward career form (w->0).

Predictors (all from season Y, so nothing leaks from Y+1):
  hist_share_logit     logit of the multi-year (EWMA) form — the anchor
  excess_vs_hist       logit(prior) - logit(hist): recent vs career; its weight
                       is the recency/mean-reversion knob
  late_share_logit     logit of the weeks>=10 share (a late-year role change)
  vacated              share freed on the Y+1 team by departed players
  competition          share claimed by arriving veterans + drafted rookies
  team_change          1 if the player switched teams
  excess_x_teamchange  movers carry less of their recent form
  team_investment      decayed draft capital — an organizational-commitment
                       prior on role (strongest where usage history is thin)
  age_c[POS], age_c2[POS]  position-specific age curve (RBs decline earliest)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.models.base import logit, sample_model, squeeze_unit

# Prior means/sds on selected slopes: share is persistent, so prior + history
# slopes together start near 1.0 (persistence), and the model adjusts from there.
_SLOPE_PRIORS = {
    "hist_share_logit": (1.0, 0.3),    # multi-year form is the persistence anchor
    "excess_vs_hist": (0.5, 0.3),      # recency weight in (0,1); <1 => mean reversion
    "excess_x_teamchange": (0.0, 0.4), # movers carry less recent form
}


@dataclass
class BetaShareModel:
    """A fitted next-season share model plus everything needed to predict."""

    target_col: str          # "next_target_share" or "next_carry_share"
    prior_col: str           # "target_share" or "carry_share"
    late_col: str            # "late_target_share" or "late_carry_share"
    vacated_col: str         # "vacated_target_share" or "vacated_carry_share"
    comp_col: str            # "incoming_comp_target" or "incoming_comp_carry"
    hist_col: str = ""       # "hist_target_share" or "hist_carry_share"
    trend_col: str = ""      # "target_trend" or "carry_trend"
    positions: list[str] = field(default_factory=list)
    predictor_names: list[str] = field(default_factory=list)
    age_mean: float = 26.0
    idata: object = None

    # ---- design matrix ---------------------------------------------------
    def _design(self, df: pd.DataFrame, fit: bool = False):
        d = df.copy()
        age = pd.to_numeric(d["age"], errors="coerce")
        if fit:
            self.age_mean = float(age.mean())
            self.positions = sorted(d["position"].unique())
        age_c = ((age.fillna(self.age_mean) - self.age_mean) / 10.0).to_numpy()
        # Persistence is anchored on the multi-year form (EWMA, which still
        # weights the latest year most), not last year alone: the two are
        # collinear and history predicts slightly better.
        prior_logit = logit(d[self.prior_col])
        hist_logit = logit(d[self.hist_col]) if self.hist_col else prior_logit
        excess = prior_logit - hist_logit  # recent form above/below career anchor
        team_change = pd.to_numeric(
            d.get("team_change", pd.Series(0, index=d.index)), errors="coerce"
        ).fillna(0.0).to_numpy()

        cols = {
            "hist_share_logit": hist_logit,
            "excess_vs_hist": excess,
            "late_share_logit": logit(d[self.late_col]),
            "vacated": _col(d, self.vacated_col),
            "competition": _col(d, self.comp_col),
            "team_change": team_change,
            "excess_x_teamchange": excess * team_change,
            # Organizational commitment (decayed draft capital): a prior on role
            # that matters most where usage history is thin.
            "team_investment": _col(d, "team_investment"),
        }
        # Position-specific age curve: age terms interacted with position dummies,
        # so an RB and a WR of the same age get different trajectories.
        for pos in self.positions:
            mask = (d["position"] == pos).to_numpy(dtype=float)
            cols[f"age_c[{pos}]"] = age_c * mask
            cols[f"age_c2[{pos}]"] = (age_c ** 2) * mask

        X = pd.DataFrame(cols)
        if fit:
            self.predictor_names = list(X.columns)
        pos_idx = pd.Categorical(d["position"], categories=self.positions).codes
        return X[self.predictor_names].to_numpy(dtype=float), pos_idx

    # ---- fit -------------------------------------------------------------
    def fit(self, transitions: pd.DataFrame, **sample_kwargs) -> "BetaShareModel":
        import pymc as pm

        df = transitions.dropna(subset=[self.target_col, self.prior_col]).copy()
        X, pos_idx = self._design(df, fit=True)
        y = squeeze_unit(df[self.target_col].to_numpy())
        n_pos, n_pred = len(self.positions), X.shape[1]

        with pm.Model() as model:
            # Position-varying intercept, partially pooled (non-centered to keep
            # NUTS geometry well-conditioned and avoid divergences).
            mu_a = pm.Normal("mu_a", 0.0, 1.5)
            sd_a = pm.HalfNormal("sd_a", 1.0)
            z_a = pm.Normal("z_a", 0.0, 1.0, shape=n_pos)
            alpha = pm.Deterministic("alpha", mu_a + z_a * sd_a)
            # Population slopes, with informative priors on a few (see
            # _SLOPE_PRIORS): prior + history slopes start near persistence.
            beta_mu = np.zeros(n_pred)
            beta_sd = np.full(n_pred, 1.0)
            for name, (m, s) in _SLOPE_PRIORS.items():
                if name in self.predictor_names:
                    i = self.predictor_names.index(name)
                    beta_mu[i], beta_sd[i] = m, s
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sd, shape=n_pred)
            phi = pm.Gamma("phi", alpha=2.0, beta=0.1)  # Beta precision

            eta = alpha[pos_idx] + pm.math.dot(X, beta)
            mu = pm.math.invlogit(eta)
            pm.Beta("obs", alpha=mu * phi, beta=(1 - mu) * phi, observed=y)

            sample_kwargs.setdefault("target_accept", 0.95)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    # ---- predict ---------------------------------------------------------
    def predict_samples(self, transitions: pd.DataFrame) -> np.ndarray:
        """Posterior samples of next-season share, shape (n_players, n_draws)."""
        unknown = set(transitions["position"].unique()) - set(self.positions)
        if unknown:
            raise ValueError(
                f"{self.target_col} model was fit on {self.positions}; got "
                f"positions {sorted(unknown)}. Filter to the model's positions "
                "first (e.g. QBs go through the pass model, not the target model)."
            )
        X, pos_idx = self._design(transitions, fit=False)
        post = self.idata.posterior
        alpha = post["alpha"].stack(s=("chain", "draw")).to_numpy()   # (n_pos, S)
        beta = post["beta"].stack(s=("chain", "draw")).to_numpy()     # (n_pred, S)
        phi = post["phi"].stack(s=("chain", "draw")).to_numpy()       # (S,)

        eta = alpha[pos_idx, :] + X @ beta                            # (n_players, S)
        mu = 1.0 / (1.0 + np.exp(-eta))
        rng = np.random.default_rng(0)
        a = np.clip(mu * phi[None, :], 1e-6, None)
        b = np.clip((1 - mu) * phi[None, :], 1e-6, None)
        return rng.beta(a, b)

    def predict_quantiles(self, transitions: pd.DataFrame,
                          qs=(0.1, 0.5, 0.9)) -> pd.DataFrame:
        samples = self.predict_samples(transitions)
        out = transitions[["player_name", "position"]].copy()
        out["pred_mean"] = samples.mean(axis=1)
        for q in qs:
            out[f"p{int(q * 100)}"] = np.quantile(samples, q, axis=1)
        return out.reset_index(drop=True)


def _col(d: pd.DataFrame, name: str) -> np.ndarray:
    """A numeric column as float, zero-filled, tolerating an absent name."""
    if not name or name not in d.columns:
        return np.zeros(len(d), dtype=float)
    return pd.to_numeric(d[name], errors="coerce").fillna(0.0).to_numpy()


# Passes, carries, and targets are modeled as separate streams — they carry
# different fantasy value, so each returning player's opportunity is projected
# per stream and recombined downstream, never collapsed into one "opportunity".

def fit_target_share(transitions: pd.DataFrame,
                     positions=("RB", "WR", "TE"), **kw) -> BetaShareModel:
    # Pass-catchers only; QBs get ~0 targets and are modeled by fit_pass_share.
    sub = transitions[transitions["position"].isin(positions)].copy()
    return BetaShareModel(
        "next_target_share", "target_share", "late_target_share",
        "vacated_target_share", "incoming_comp_target",
        hist_col="hist_target_share", trend_col="target_trend",
    ).fit(sub, **kw)


def fit_carry_share(transitions: pd.DataFrame, positions=("RB",), **kw) -> BetaShareModel:
    # Carries concentrate in the backfield; restrict to RB for now (QB rushing
    # is a separate regime, a documented future addition).
    sub = transitions[transitions["position"].isin(positions)].copy()
    return BetaShareModel(
        "next_carry_share", "carry_share", "late_carry_share",
        "vacated_carry_share", "incoming_comp_carry",
        hist_col="hist_carry_share", trend_col="carry_trend",
    ).fit(sub, **kw)


def fit_pass_share(transitions: pd.DataFrame, positions=("QB",), **kw) -> BetaShareModel:
    # Share of team pass attempts a QB threw. There are no vacated/competition
    # pass columns yet, so those predictors default to zero; persistence (a
    # QB keeping the starting job) plus age carry the model. Bimodal starter vs
    # backup, so treat as a coarse v1.
    sub = transitions[transitions["position"].isin(positions)].copy()
    return BetaShareModel(
        "next_pass_share", "pass_share", "pass_share",  # no late-pass split; reuse level
        "vacated_pass_share", "incoming_comp_pass",     # absent -> zero-filled
        hist_col="hist_pass_share", trend_col="pass_trend",
    ).fit(sub, **kw)
