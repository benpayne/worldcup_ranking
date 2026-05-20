"""Generate the human-readable model report."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig
from .ranking import RankingResult


_CAVEATS = [
    "FIFA's official ranking is canonical but not built primarily as a predictive model.",
    "Elo is a strong baseline but ignores roster turnover, injuries, and tactical context.",
    "Public xG and squad-value data are inconsistent across national teams; this model "
    "uses a goal-based proxy unless a squad CSV is supplied.",
    "The output is a *predictive strength ranking* of pre-tournament form. It is not a "
    "guarantee of tournament finish - upsets are intrinsic to knockout football.",
]


def render_report(
    result: RankingResult, config: AppConfig, output_dir: str | Path
) -> Path:
    """Render ``outputs/model_report.md`` describing the run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_report.md"

    weights_table = "\n".join(
        f"| {k} | {v:.4f} |" for k, v in result.weights_effective.items()
    )
    base_table = "\n".join(
        [
            f"| elo | {config.weights.elo:.4f} |",
            f"| recent_form | {config.weights.recent_form:.4f} |",
            f"| goal_performance | {config.weights.goal_performance:.4f} |",
            f"| squad_strength | {config.weights.squad_strength:.4f} |",
        ]
    )

    sources = "\n".join(
        f"- **{s.name}** - {s.url} ({s.licence}). {s.notes}".rstrip()
        for s in result.sources
    )

    top_table_rows = []
    head = result.rankings.head(config.output.top_n)
    for _, row in head.iterrows():
        squad = row["squad_strength_score"]
        squad_str = "n/a" if squad != squad else f"{squad:.1f}"  # NaN-safe
        top_table_rows.append(
            f"| {int(row['rank'])} | {row['team']} | {row['final_score']:.2f} | "
            f"{row['elo_score']:.1f} | {row['recent_form_score']:.1f} | "
            f"{row['goal_performance_score']:.1f} | {squad_str} | "
            f"{int(row['matches_used']) if row['matches_used'] == row['matches_used'] else 0} |"
        )
    top_table = "\n".join(top_table_rows)

    notes_block = (
        "\n".join(f"- {n}" for n in result.notes) if result.notes else "- None recorded."
    )
    caveats_block = "\n".join(f"- {c}" for c in _CAVEATS)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = f"""# World Cup Strength Ranking - Model Report

_Generated: {generated}_

## Tournament

- Start date (cutoff): **{result.cutoff.date()}**
- Form look-back window: **{config.tournament.form_window_days} days**
- Minimum matches required for full confidence: **{config.tournament.min_matches}**

## Data

- Match results loaded: **{result.matches_total:,}**
- Matches used (strictly before cutoff): **{result.matches_in_window:,}**
- Teams ranked: **{result.teams_ranked}**

### Sources

{sources}

## Weights

### Configured

| component | weight |
|-----------|-------:|
{base_table}

### Effective (after dropping missing components and renormalizing)

| component | weight |
|-----------|-------:|
{weights_table}

## Top {config.output.top_n}

| rank | team | final_score | elo | recent_form | goal_perf | squad | matches_used |
|-----:|------|-----------:|----:|-----------:|---------:|------:|------------:|
{top_table}

## Missing-data handling

{notes_block}

## Method summary

1. **Elo** ratings are computed by walking every international match strictly
   *before* the cutoff date, using
   `expected = 1 / (1 + 10^(-rating_diff / 400))` with a Davidson-style
   goal-difference multiplier and per-tournament importance weights.
2. **Recent form** is the exponentially-decayed average of
   `result - expected` over the look-back window (half-life
   {int(config.recent_form.half_life_days)} days,
   opponent-adjusted via the running Elo).
3. **Goal performance** is a capped goals-for / goals-against index over the
   same window.
4. **Squad strength** is loaded from a user-supplied CSV when available; when
   absent the component is dropped and the remaining weights are
   renormalized.
5. All components are min-max rescaled to 0-100, then combined with the
   effective weights to produce `final_score`.

## Caveats

{caveats_block}
"""
    path.write_text(content, encoding="utf-8")
    return path
