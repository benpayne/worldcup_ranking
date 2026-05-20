# worldcup-ranker

Pre-tournament strength ranking for the men's FIFA World Cup, computed
entirely from **free and public** data. Given a tournament start date, the
tool produces a ranked list of the top 24 (or top N) qualified teams using
a transparent, weighted model:

```
Team Score = 0.50 * Elo                      (strength)
           + 0.25 * Recent form              (results vs. expectation, decayed)
           + 0.15 * Goal-based performance   (capped GF / GA proxy)
           + 0.10 * Squad strength (optional)
```

If the optional squad-strength component is unavailable the remaining
weights are renormalized and the omission is reported in
`outputs/model_report.md`.

## Project layout

```
worldcup_ranker/
  pyproject.toml
  README.md
  config/default.yaml
  data/
    raw/                  # source CSVs go here
    processed/            # intermediate derivatives (cache)
  outputs/                # generated rankings, report, chart
  src/worldcup_ranker/
    __init__.py
    cli.py
    config.py
    data_sources.py
    elo.py
    features.py
    ranking.py
    report.py
    plotting.py
  tests/
    test_elo.py
    test_features.py
    test_ranking.py
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11+ is required.

## Quick start

```bash
# 1. Get the public match-results CSV (martj42, CC0).
worldcup-ranker fetch-data
# ... or download manually from
# https://github.com/martj42/international_results/raw/master/results.csv
# and place it at data/raw/results.csv.

# 2. Run the ranking with the default cutoff (2026-06-11).
worldcup-ranker rank

# 3. View outputs.
cat outputs/top_24_rankings.csv
open outputs/top_24.png          # or xdg-open / start
less outputs/model_report.md
```

### Useful flags

```bash
worldcup-ranker rank \
    --start-date 2026-06-11 \
    --qualified-teams config/wc2026_teams.csv \
    --results-csv data/raw/results.csv \
    --squad-csv data/raw/squad_strength.csv \
    --config config/default.yaml
```

- `--qualified-teams` is a CSV with a `team` column. The default config
  points at `config/wc2026_teams.csv`, a bundled best-guess list of the
  2026 World Cup field that excludes non-FIFA sides (Basque Country,
  Catalonia, etc.) and currently-suspended teams (e.g. Russia, banned
  from FIFA competitions since Feb 2022). Replace it with the confirmed
  48-team list once FIFA publishes it, or set
  `data.qualified_teams_csv: null` in your config to rank every team
  that appears in the match data.
- `--squad-csv` is a CSV with columns `team,score` (any numeric scale). When
  omitted the squad component is dropped and the remaining weights are
  renormalized.
- `--no-chart` skips PNG generation.

### Re-render only the report

```bash
worldcup-ranker report
```

## Data sources

| Component       | Source                                                 | Licence |
|-----------------|--------------------------------------------------------|---------|
| Match results   | [martj42/international_results](https://github.com/martj42/international_results) | CC0 1.0 |
| Elo             | Computed in-process from the match results above       | n/a     |
| Recent form     | Same as above (opponent-adjusted via Elo)              | n/a     |
| Goal proxy      | Same as above; optionally enriched with StatsBomb xG   | n/a     |
| xG enrichment   | [statsbomb/open-data](https://github.com/statsbomb/open-data) (optional) | StatsBomb OD User Agreement (non-commercial, attribution) |
| Squad strength  | User-supplied CSV, or built from StatsBomb open-data   | varies  |

The Kaggle mirror of the same dataset is at
<https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017>
and is also CC0. The CLI fetches the file directly from the upstream
GitHub repo so no Kaggle credentials are required.

## Model details

### Elo (50%)

`expected = 1 / (1 + 10 ** (-rating_diff / 400))` with:

- configurable initial rating (default 1500)
- configurable home advantage (default +65, disabled at neutral venues)
- per-tournament importance multipliers (Friendly = 1.0, ..., World Cup = 4.0)
- Davidson-style goal-difference multiplier (`G(1)=1`, `G(2)=1.5`,
  `G(n>=3)=(11+n)/8`)

Elo is computed by walking the entire pre-cutoff history of international
matches. **No post-cutoff match is ever used.** Tests in
`tests/test_elo.py` and `tests/test_ranking.py` assert this.

### Recent form (25%)

For each match in the look-back window (default 730 days):

1. Compute the team's `result` (1 / 0.5 / 0).
2. Compute the team's `expected` outcome from the *pre-match* Elo state
   (when `opponent_adjusted=True`) or 0.5 otherwise.
3. Multiply `(result - expected)` by an exponential decay weight
   `0.5 ** (age_days / half_life_days)` (default half-life: 365 days).

The team's recent-form raw score is the weighted mean of those signed
deltas (roughly in `[-1, 1]`), later min-max rescaled to 0-100.

### Goal performance (15%)

Capped goals-for and goals-against averages over the look-back window:

```
goal_perf_raw = attack_weight  * mean(min(goals_for,  cap))
              - defense_weight * mean(min(goals_against, cap))
```

**Optional xG enrichment:** when `--xg-csv` is supplied (typically
produced by `worldcup-ranker statsbomb build`), each covered match
uses `xG - xGA` (still capped) in place of the goals proxy. See the
"StatsBomb xG and squad enrichment" section below for details.

### Small-sample handling

By default, teams with fewer than `tournament.min_matches` matches in
the look-back window are **dropped** from the output. This prevents a
single 4-0 friendly from floating a team to the top of the table by
anchoring the min-max rescale (the bug that put Basque Country on the
first published run). To keep them in the table with a `"<N matches"`
note instead, set `tournament.drop_below_min_matches: false`.

### StatsBomb xG and squad enrichment (optional)

`worldcup-ranker statsbomb build` reads a local clone of
[`statsbomb/open-data`](https://github.com/statsbomb/open-data) — free
for non-commercial use with attribution under the StatsBomb Open Data
User Agreement — and emits two CSVs:

- `data/processed/statsbomb_xg.csv` — per-match xG (`home_xg`,
  `away_xg`) for every men's national-team tournament covered by the
  open dataset (recent World Cups, Euros, Copa América).
- `data/processed/statsbomb_squad.csv` — per-team squad-strength
  scores, computed by summing each player's xG (shots taken) + xA
  proxy (shots they key-passed) across matches in the look-back
  window, then taking the top 23 contributors per nation.

Hook both into the ranking:

```bash
git clone --depth 1 https://github.com/statsbomb/open-data ~/sbod
worldcup-ranker statsbomb build --open-data-path ~/sbod
worldcup-ranker rank \
    --xg-csv   data/processed/statsbomb_xg.csv \
    --squad-csv data/processed/statsbomb_squad.csv
```

For matches covered by StatsBomb, the goal-performance feature uses
`xG - xGA` (capped) in place of the goals-based proxy. Matches
outside coverage fall through to the existing capped-goals path, so
no team is penalised for tournaments StatsBomb hasn't released.

We deliberately do **not** use the `statsbombpy` PyPI package — it is
designed for the commercial API and has no first-class mode for
reading a local clone, while the open-data JSON schema is small
enough that the in-tree reader is ~100 lines.

**Coverage caveat:** the StatsBomb open data only covers a handful of
men's national-team tournaments (WC 2018, WC 2022, Euro 2020, Euro
2024, Copa América 2024). Teams that haven't featured in those events
contribute 0 to the squad score, so the squad component will
under-weight strong sides from confederations that aren't in the
covered tournaments. The xG path degrades more gracefully because of
the goals-based fallback.

### Squad strength (10%, optional)

If a CSV is supplied (`--squad-csv`), its `score` column is min-max
rescaled to 0-100. Otherwise the component is dropped and the remaining
weights are renormalized to sum to 1.0 - this is reported in
`outputs/model_report.md`.

We deliberately do **not** scrape Transfermarkt or other proprietary
sources. If you want squad-value-based scores you can build the CSV
yourself from any source whose terms permit it.

## Outputs

After a successful `rank`:

- `outputs/top_24_rankings.csv` - the top N (default 24) qualified teams.
- `outputs/full_rankings.csv` - every team scored.
- `outputs/model_report.md` - cutoff date, match counts, effective
  weights, top-N table, missing-data handling, caveats.
- `outputs/top_24.png` - horizontal bar chart (skippable via `--no-chart`).

Columns in the ranking CSVs:

| column | meaning |
|---|---|
| `rank` | 1 = best |
| `team` | canonical team name (after alias mapping) |
| `final_score` | weighted 0-100 score |
| `elo_score`, `recent_form_score`, `goal_performance_score`, `squad_strength_score` | component sub-scores (0-100; squad may be empty) |
| `matches_used` | matches in the look-back window |
| `last_match_date` | last pre-cutoff match used |
| `data_sources` | provenance string |
| `notes` | warnings (e.g. small sample) |

## Tests

```bash
pytest -q
```

Notable invariants under test:

- Date filtering is strict (`matches['date'] < cutoff`).
- Post-cutoff matches do not affect any output.
- Home advantage is disabled at neutral venues.
- Tournament importance multipliers amplify rating swings.
- Capping caps blow-out goal differences.
- When the squad CSV is missing the weights are renormalized.

## Continuous publication (GitHub Actions + Pages)

`.github/workflows/rank.yml` runs the ranking pipeline and publishes the
result to **GitHub Pages** at
`https://<owner>.github.io/<repo>/`.

Triggers:

- **Weekly cron** - Mondays at 06:00 UTC (martj42's dataset refreshes
  roughly weekly).
- **Push to `main`** when source, config, scripts, or the workflow file
  itself change.
- **Manual** via the Actions tab (`workflow_dispatch`), with an optional
  `start_date` input so you can re-rank for a different cutoff.

The workflow installs the package with the `[site]` extra, runs the test
suite as a guard, calls `worldcup-ranker fetch-data` and
`worldcup-ranker rank`, then runs `scripts/build_site.py` to assemble a
self-contained `site/` directory (HTML report + embedded chart +
downloadable CSVs) which is published with `actions/deploy-pages`.

### One-time setup

In the repo settings:

1. **Settings -> Pages -> Build and deployment -> Source**: select
   **GitHub Actions**.
2. **Settings -> Actions -> General -> Workflow permissions**: leave at
   the default ("Read repository contents"); the workflow declares the
   extra `pages: write` and `id-token: write` permissions it needs.

After the first successful run the deployed URL is printed in the
workflow summary and remains stable for subsequent runs.

### Local preview

```bash
pip install -e ".[site]"
worldcup-ranker fetch-data
worldcup-ranker rank
python scripts/build_site.py
python -m http.server --directory site 8000
# open http://localhost:8000
```

## Caveats

- FIFA's official ranking is canonical but not designed primarily as a
  predictive model.
- Elo is a strong baseline but ignores roster turnover, injuries, and
  tactical context.
- Public xG / market-value data for national teams is patchy; the
  goal-based proxy is a pragmatic stand-in.
- This is a *predictive strength ranking*, not a guarantee of tournament
  finish. Knockout football is intrinsically noisy.
