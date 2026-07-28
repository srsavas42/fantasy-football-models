"""Dirichlet-Multinomial team allocation — the joint volume model.

Where `volume_season.BetaShareModel` projects each returning player's share
independently, this models a whole team-position roster as one simplex: for each
(team, next-season) group the players' next-season opportunity counts follow a
Dirichlet-Multinomial whose per-player concentration is a softmax of the same
usage/age/investment features. Two things fall out of the structure for free and
so are *not* covariates here:

  * **competition** — adding a claimant to the group dilutes everyone via the
    softmax, so incoming veterans and rookies compete automatically;
  * **vacated opportunity** — a departed player is simply absent from the Y+1
    group, and their share redistributes.

Shares therefore sum to 1 within a team by construction, and rookies (zero usage
history) sit in the group carried by their draft capital and an is_rookie term.

The ragged groups are handled by flattening every (group, player) into rows and
using a fixed group-membership matrix G for the segmented softmax, with the
Dirichlet-Multinomial log-likelihood added as a `pm.Potential`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.models.base import sample_model

_EPS = 0.02  # share floor so log() is finite for rookies (hist_share == 0)

# Informative slope priors: the history anchor is ~proportional allocation.
_SLOPE_PRIORS = {
    "log_hist": (1.0, 0.4),
    "excess": (0.5, 0.4),
}


def _logshare(s: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(s, dtype=float), 0.0, 1.0) + _EPS)


@dataclass
class DirichletAllocation:
    """A fitted team-allocation model plus what's needed to predict."""

    positions: list[str] = field(default_factory=list)
    predictor_names: list[str] = field(default_factory=list)
    age_mean: float = 26.0
    idata: object = None

    # ---- design ----------------------------------------------------------
    def _design(self, df: pd.DataFrame, fit: bool = False):
        d = df.copy()
        age = pd.to_numeric(d.get("age"), errors="coerce")
        if fit:
            self.age_mean = float(age.mean())
            self.positions = sorted(d["position"].unique())
        age_c = ((age.fillna(self.age_mean) - self.age_mean) / 10.0).to_numpy()

        log_hist = _logshare(d["hist_share"])
        excess = _logshare(d["prior_share"]) - log_hist
        cols = {
            "log_hist": log_hist,
            "excess": excess,
            "late": _logshare(d["late_share"]),
            "is_rookie": _num(d, "is_rookie"),
            "team_change": _num(d, "team_change"),
            "draft_value": _num(d, "draft_value"),
            "draft_value_x_years": _num(d, "draft_value") * (_num(d, "years_since_draft") / 5.0),
            "contract_value": _num(d, "contract_value"),
            "contract_value_x_year": _num(d, "contract_value") * (_num(d, "contract_year") / 3.0),
        }
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
    def fit(self, groups: pd.DataFrame, **sample_kwargs) -> "DirichletAllocation":
        import pymc as pm
        import pytensor.tensor as pt

        d = groups.reset_index(drop=True)
        X, pos_idx = self._design(d, fit=True)
        counts = d["label_count"].to_numpy(dtype=float)
        G = _group_matrix(d["group_id"].to_numpy())  # (n_groups, n_rows)
        n_pos, n_pred = len(self.positions), X.shape[1]

        beta_mu = np.zeros(n_pred)
        beta_sd = np.ones(n_pred)
        for name, (m, s) in _SLOPE_PRIORS.items():
            if name in self.predictor_names:
                i = self.predictor_names.index(name)
                beta_mu[i], beta_sd[i] = m, s

        N_group = G @ counts

        with pm.Model() as model:
            pos_base = pm.Normal("pos_base", 0.0, 1.5, shape=n_pos)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sd, shape=n_pred)
            phi = pm.Gamma("phi", alpha=2.0, beta=0.02)  # DM concentration (mean ~100)

            eta = pos_base[pos_idx] + pm.math.dot(X, beta)
            exp_eta = pt.exp(eta - pt.max(eta))
            denom = pt.dot(G.T, pt.dot(G, exp_eta))  # per-row group sum
            p = exp_eta / denom
            alpha = phi * p

            logp = (
                (pt.gammaln(phi) - pt.gammaln(N_group + phi)).sum()
                + (pt.gammaln(counts + alpha) - pt.gammaln(alpha)).sum()
            )
            pm.Potential("dm", logp)

            sample_kwargs.setdefault("target_accept", 0.9)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    # ---- predict ---------------------------------------------------------
    def predict_shares(self, groups: pd.DataFrame) -> np.ndarray:
        """Posterior samples of each player's within-group share, (n_rows, n_draws).

        Softmax is renormalized within each group per posterior draw, so columns
        sum to 1 within a group_id.
        """
        d = groups.reset_index(drop=True)
        X, pos_idx = self._design(d, fit=False)
        G = _group_matrix(d["group_id"].to_numpy())
        post = self.idata.posterior
        pos_base = post["pos_base"].stack(s=("chain", "draw")).to_numpy()  # (n_pos, S)
        beta = post["beta"].stack(s=("chain", "draw")).to_numpy()          # (n_pred, S)

        eta = pos_base[pos_idx, :] + X @ beta          # (n_rows, S)
        eta = eta - eta.max(axis=0, keepdims=True)
        exp_eta = np.exp(eta)
        denom = G.T @ (G @ exp_eta)                    # (n_rows, S)
        return exp_eta / denom

    def predict_quantiles(self, groups: pd.DataFrame, qs=(0.1, 0.5, 0.9)) -> pd.DataFrame:
        s = self.predict_shares(groups)
        out = groups.reset_index(drop=True)[["group_id", "player_name", "position"]].copy()
        out["pred_share"] = s.mean(axis=1)
        for q in qs:
            out[f"p{int(q * 100)}"] = np.quantile(s, q, axis=1)
        return out


def _num(d: pd.DataFrame, name: str) -> np.ndarray:
    if name not in d.columns:
        return np.zeros(len(d), dtype=float)
    return pd.to_numeric(d[name], errors="coerce").fillna(0.0).to_numpy()


def _group_matrix(group_ids: np.ndarray) -> np.ndarray:
    """One-hot (n_groups, n_rows) membership matrix for segmented sums."""
    codes, _ = pd.factorize(group_ids)
    n_groups, n_rows = codes.max() + 1, len(codes)
    G = np.zeros((n_groups, n_rows), dtype=float)
    G[codes, np.arange(n_rows)] = 1.0
    return G


def fit_allocation(seasons, resource: str = "target", source: str = "auto",
                   **kw) -> tuple["DirichletAllocation", pd.DataFrame]:
    """Convenience: build groups for a resource and fit the allocation."""
    from ffmodel.features.crossseason import build_team_groups

    groups = build_team_groups(seasons, resource=resource, source=source)
    return DirichletAllocation().fit(groups, **kw), groups
