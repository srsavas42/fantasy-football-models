# Fantasy Football Distributional Modeling

Statistical models that produce **distributions** of fantasy football outcomes — not point estimates — on both season-long and weekly horizons, supporting three pillars:

1. **Draft value** — tier gaps, pre-season expected value, and mid-draft positional trade-offs.
2. **Volume prediction** — opportunity is king; predict each player's share of team plays.
3. **Weekly outcomes** — per-week outcome distributions for start/sit and lineup optimization.

## Architecture

Fantasy points for a player-week are simulated bottom-up, with each layer a hierarchical Bayesian model (PyMC):

```
team plays & pass rate  →  opportunity share  →  per-touch efficiency  →  scoring
   (NegBinom/Binomial)     (Dirichlet-Multinomial      (hierarchical         (PPR /
                            over the active roster)     Normal/Poisson)       Half / Std)
```

Sampling all layers over posterior draws yields the full outcome distribution; season projections aggregate simulated weeks. Because opportunity shares renormalize over whoever is *active*, a starter's injury automatically flows volume to backups.

Key modeling choices:

- **Empirical roles over listed depth charts.** Role tiers come from EWMA trailing snap share (route participation where available); listed depth charts + ADP/ECR are only a cold-start fallback for week 1, rookies, and team changes.
- **Efficiency feeds volume.** Trailing per-opportunity efficiency (yds/route-run, yds/touch) enters the share model — coaches route opportunity to efficient players.
- **Partial pooling everywhere.** Small-sample players shrink toward position-level priors.
- **Calibration is the acceptance gate.** Walk-forward backtests score CRPS/log-score against prior-season-PPG and ECR baselines, with PIT/coverage checks that intervals are honest.

## Package

Code lives in an installable package under `src/ffmodel/`:

```
src/ffmodel/
  config.py       scoring rules (verified against this repo's CSVs), paths, season coverage
  data/           hybrid data layer:
    schema.py       canonical player-week schema shared by every source
    ingest.py       nflverse via nflreadpy, parquet-cached (weekly, PBP, snaps, depth charts,
                    injuries, schedules, rosters, id map)
    legacy.py       the CSVs committed to this repo (weekly 1999-2021, yearly 1970-2021,
                    snapcounts 2013-2020, FantasyPros ADP/ECR)
    loaders.py      load_player_weeks(seasons) — one call, one schema, auto source fallback
    teams.py        canonical franchise codes (relocations collapse: STL/LA→LAR, OAK→LV)
    identity.py     canonical gsis player dimension + cross-provider id joins
    cfbd.py, coaching.py, odds.py, weather.py, sleeper.py   external sources
  features/         raw stat lines → model-ready covariates:
    trailing.py     the one leak-free EWMA builder (shift(1) then EWMA)
    volume.py       team totals, usage shares; snaps.py  snap-share integration
    crossseason.py  season usage, career history, vacated/competition,
                    build_transitions (per-player) and build_team_groups (roster groups)
    investment.py   draft capital; contracts.py  veteran contract commitment
  models/
    volume_season.py  hierarchical Beta — per-player next-season share
    volume_alloc.py   Dirichlet-Multinomial — joint team allocation (shares sum to 1)
  projections/
    season_volume.py  next-season share distributions + breakout report
  simulation/
    scoring.py      stat line → fantasy points (reproduces the CSV point columns exactly)
```

### Quickstart

```bash
pip install -e ".[dev]"        # add ".[models]" for pymc/arviz when fitting
pytest                          # network-free test suite

python -c "
from ffmodel.data import load_player_weeks
df = load_player_weeks([2019, 2020])
print(df.head())
"
```

`load_player_weeks` tries nflverse first (richer: player ids, real targets, 18-week seasons kept current) and falls back to the committed CSVs per season when offline.

### Data acquisition

The provider-aware data CLI caches parquet plus provenance manifests and keeps
mutable inputs as immutable `as_of` snapshots:

```bash
ffmodel-data doctor
ffmodel-data bootstrap --seasons 2022 2023 2024 2025
ffmodel-data nflverse --seasons 2022 2023 2024 2025 --datasets pbp
ffmodel-data sleeper
```

CollegeFootballData reads its credential from the Git-ignored project `.env`,
caches every response as Parquet, and enforces local/per-run quota guards. The
Odds API is intentionally deferred. Open-Meteo and Sleeper require no key.
Setup, scheduling, licensing, and point-in-time backtest instructions are in
[docs/data-sources.md](docs/data-sources.md).

Wikipedia HC/OC assignments and coach lineage use a separate, resumable pull:

```bash
pip install -e ".[scrape]"
ffmodel-coaches                       # every committed team-season, 1970-2021
ffmodel-coaches --seasons 2018:2025  # add nflverse-backed recent seasons
```

The command archives exact MediaWiki revisions in the ignored cache and writes
source-attributed assignment, career-history, selected scheme-source, lineage,
and review tables under `data/coaching/wikipedia/`. Wikipedia job titles are not
proof of play-calling responsibility; confirmed effective-date overrides remain
in `data/manual/coach_team_period.csv`. See the coaching section of the data
source guide before using the lineage as a model prior.

### Cross-season volume & breakout report (Phase 3A)

```python
from ffmodel.features import crossseason as cs
from ffmodel.models import volume_season as vs
from ffmodel.projections import season_volume as sv

trans = cs.build_transitions(range(2015, 2021), source="legacy")   # returning players, Y->Y+1
train, test = trans[trans.transition < "2019->2020"], trans[trans.transition == "2019->2020"]

target_model = vs.fit_target_share(train)     # hierarchical Beta (needs the ".[models]" extra)
carry_model  = vs.fit_carry_share(train)
sv.breakout_report(test, target_model, carry_model, threshold=0.05)  # ranked P(volume uptick)
```

The Beta share model is centered on year-over-year persistence (share is sticky) and adjusts for both sides of the opportunity ledger:

- **Vacated opportunity** — volume freed when teammates leave (from roster diffs).
- **Incoming competition** — volume claimed by players *arriving* at the same position: signed/traded veterans (their prior-team share) and drafted rookies (draft capital, `features/draft.py`). This is the other half — freed targets mean little if the team also signed a star and drafted a receiver.

Modeling competition matters: for RBs the competition coefficient is strongly negative and it *unmasks* the vacated-opportunity signal (its coefficient roughly 6× larger once competition is controlled for). Net opportunity (vacated − competition) tracks realized carry-share change far better than vacated alone (Spearman ~0.26 vs ~0.03) — see `scripts/validate_crossseason.py`. It roughly matches a persistence baseline on point error but adds calibrated ~80% intervals and per-player breakout probabilities. v1 covers returning players as the subjects (rookies enter only as competition, not yet as projected players); the veteran-competition proxy and rookie draft data use the offline combine file, upgraded to nflverse draft picks when online.

### Joint team allocation (the volume model)

Projecting each player's share independently leaves shares that don't sum to 1 on a team. The allocation model treats each team-position roster as one simplex:

```python
from ffmodel.features import crossseason as cs
from ffmodel.models import volume_alloc as va

groups = cs.build_team_groups(range(2015, 2021), resource="target")  # or "carry" / "pass"
model = va.DirichletAllocation().fit(groups)
model.predict_quantiles(groups)          # per-player share, P10/P50/P90, summing to 1 per team
```

Next-season opportunity counts follow a Dirichlet-Multinomial whose per-player concentration is a softmax over usage history, position age curves, and team investment. Two things become **structural** rather than covariates: **competition** (adding a claimant dilutes everyone through the softmax) and **vacated opportunity** (a departed player is simply absent from the group). Rookies sit in the group with zero usage history, carried by draft capital — the socket a full rookie model drops into later. Targets, carries, and passes are modeled as separate resources since each carries different fantasy value.

**Team investment** (`features/investment.py`, `features/contracts.py`) supplies the organizational-commitment prior: the model is given raw draft value and a `draft_value × years_since_draft` interaction and *learns* the decay (fitted ≈ +0.48 / −0.21, a gentler fade than a hand-picked rate), plus veteran contract size and its fade over the deal.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Package scaffolding, config, scoring, tests | ✅ |
| 1 | Hybrid data layer (nflverse + legacy CSVs, parquet cache) | ✅ |
| 2 | Features: usage shares, empirical role tiers, trailing efficiency, game script, active-set/injury logic | ✅ |
| 3A | **Cross-season volume** (year-over-year share via hierarchical Beta) + breakout report | ✅ |
| 3A+ | **Joint team allocation** (Dirichlet-Multinomial over each team-position roster) + team investment (draft capital, contracts) | ✅ |
| 3B | Within-season **volume models** (team plays/pass rate + weekly Dirichlet share) | next |
| 4 | Efficiency models (yds/touch, TD, catch rate) | |
| 5 | Simulation engine: posterior predictive → weekly & season point distributions | |
| 6 | Evaluation: walk-forward backtests, CRPS/log-score, calibration | |
| 7 | Weekly pillar: start/sit lineup optimization | |
| 8 | Draft pillar: tiers, pre-season EV, positional trade-offs | |
| 9 | Alt-data signal layer: BlueSky/news → live role-prior adjustments (not backtestable, so live-only) | |

# Fantasy Football Data Sets

This repo began as a fork of [fantasydatapros/data](https://github.com/fantasydatapros/data); the CSVs below remain available and power the legacy loaders and offline tests.

## Strength of Schedule data
Strength of Schedule data is available in the sos directory. Data is available going back to 1999. To load this data in pandas using the following the following url format:
https://raw.githubusercontent.com/fantasydatapros/data/master/sos/{year}.csv

For example, in pandas do the following:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/sos/1999.csv', index_col=0)
    df.index = df.index.rename('Team')

## Weekly Fantasy Stats
Weekly stats going back to 1999 are available are exposed through the following url format

https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/{year}/week{week}.csv

To grab weekly data for year 2019, week 1 in pandas, you would do:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/2019/week1.csv')

## Yearly Fantasy stats
Yearly fantasy stats are available going back to 1970.

The url format:
https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/{year}.csv

To grab yearly data for 2019 in pandas, do the following:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2019.csv')
