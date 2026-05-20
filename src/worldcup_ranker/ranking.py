"""End-to-end ranking pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import AppConfig
from .data_sources import (
    PRIMARY_SOURCE,
    DataSourceInfo,
    filter_before,
    load_qualified_teams,
    load_results,
    load_squad_strength,
)
from .elo import EloState, compute_elo, elo_for_teams
from .features import (
    compute_goal_performance,
    compute_recent_form,
    compute_squad_strength,
    minmax_rescale,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class RankingResult:
    """Bundle returned by :func:`run_ranking` for downstream reporting."""

    rankings: pd.DataFrame
    weights_effective: dict[str, float]
    cutoff: pd.Timestamp
    matches_total: int
    matches_in_window: int
    teams_ranked: int
    notes: list[str] = field(default_factory=list)
    sources: list[DataSourceInfo] = field(default_factory=list)
    elo_state: Optional[EloState] = None


def _effective_weights(
    base: dict[str, float], present_components: set[str]
) -> dict[str, float]:
    """Drop missing components and renormalize the remaining weights to 1.0."""
    filtered = {k: v for k, v in base.items() if k in present_components and v > 0}
    total = sum(filtered.values())
    if total == 0:
        raise ValueError("All scoring components are missing or zero-weighted.")
    return {k: v / total for k, v in filtered.items()}


def run_ranking(config: AppConfig) -> RankingResult:
    """Execute the full ranking pipeline.

    Steps:
      1. Load match results and apply team-name aliases.
      2. Filter matches strictly before the tournament start date.
      3. Compute Elo from the full pre-cutoff history.
      4. Compute recent-form and goal-performance features over the look-back
         window.
      5. Optionally load a squad-strength CSV.
      6. Min-max rescale each feature to 0-100 and combine using configured
         weights (renormalized if a component is missing).
    """
    aliases = config.data.team_aliases or {}
    matches_all = load_results(config.data.results_csv, team_aliases=aliases)
    matches_total = len(matches_all)

    cutoff = pd.Timestamp(config.tournament.start_date)
    matches = filter_before(matches_all, cutoff)
    LOGGER.info(
        "Loaded %d matches total; %d before cutoff %s",
        matches_total,
        len(matches),
        cutoff.date(),
    )

    if matches.empty:
        raise ValueError(
            f"No matches found before cutoff {cutoff.date()!s}; check results CSV."
        )

    # 1. Elo over the entire pre-cutoff history.
    elo_state = compute_elo(matches, config.elo)

    # 2. Determine candidate teams.
    qualified = load_qualified_teams(
        config.data.qualified_teams_csv, team_aliases=aliases
    )
    if qualified:
        teams = qualified
    else:
        teams = sorted(elo_state.ratings.keys())

    # 3. Features in look-back window.
    form_df = compute_recent_form(
        matches,
        cutoff=cutoff,
        window_days=config.tournament.form_window_days,
        config=config.recent_form,
        elo_state=elo_state,
    )
    goals_df = compute_goal_performance(
        matches,
        cutoff=cutoff,
        window_days=config.tournament.form_window_days,
        config=config.goal_performance,
    )

    # 4. Squad strength (optional).
    squad_raw = load_squad_strength(config.squad_strength.csv_path, aliases)
    squad_df = compute_squad_strength(squad_raw, teams, config.squad_strength)

    # 5. Build a candidate frame keyed on team.
    base = pd.DataFrame({"team": teams})
    elo_df = elo_for_teams(elo_state, teams).rename(
        columns={"matches_played": "matches_played_total"}
    )
    df = base.merge(elo_df, on="team", how="left")
    # Drop the form/goal last_match_date to avoid colliding with elo_df.
    form_df_merge = form_df.drop(columns=["last_match_date"], errors="ignore")
    df = df.merge(form_df_merge, on="team", how="left")
    df = df.merge(goals_df, on="team", how="left")
    if squad_df is not None:
        df = df.merge(squad_df, on="team", how="left")

    # 6. Flag teams with too few matches in the window (record a note).
    notes: list[str] = []
    min_matches = config.tournament.min_matches
    low_sample_teams = set(
        df.loc[df["matches_used"].fillna(0) < min_matches, "team"].tolist()
    )
    if low_sample_teams:
        notes.append(
            f"{len(low_sample_teams)} team(s) have fewer than {min_matches} matches "
            f"in the {config.tournament.form_window_days}-day window: "
            f"{', '.join(sorted(low_sample_teams))}. They are still ranked but their "
            "form/goal scores rely on a small sample."
        )

    # 7. Rescale features to 0-100.
    df["elo_score"] = minmax_rescale(df["elo"])
    df["recent_form_score"] = minmax_rescale(df["recent_form_raw"])
    df["goal_performance_score"] = minmax_rescale(df["goal_perf_raw"])

    present = {"elo", "recent_form", "goal_performance"}
    if squad_df is not None and "squad_raw" in df.columns:
        df["squad_strength_score"] = minmax_rescale(df["squad_raw"])
        present.add("squad_strength")
    else:
        df["squad_strength_score"] = np.nan
        notes.append(
            "Squad-strength data not provided; component omitted and remaining "
            "weights renormalized."
        )

    base_weights = {
        "elo": config.weights.elo,
        "recent_form": config.weights.recent_form,
        "goal_performance": config.weights.goal_performance,
        "squad_strength": config.weights.squad_strength,
    }
    eff = _effective_weights(base_weights, present)

    score_components = {
        "elo": "elo_score",
        "recent_form": "recent_form_score",
        "goal_performance": "goal_performance_score",
        "squad_strength": "squad_strength_score",
    }
    df["final_score"] = 0.0
    for component, weight in eff.items():
        col = score_components[component]
        df["final_score"] = df["final_score"] + df[col].fillna(0.0) * weight

    # 8. Decorate with metadata.
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df["data_sources"] = PRIMARY_SOURCE.name
    df["notes"] = ""
    if low_sample_teams:
        df.loc[df["team"].isin(low_sample_teams), "notes"] = (
            f"<{min_matches} matches in window"
        )

    df = df.rename(columns={"matches_used": "matches_used"})

    columns = [
        "rank",
        "team",
        "final_score",
        "elo_score",
        "recent_form_score",
        "goal_performance_score",
        "squad_strength_score",
        "matches_used",
        "last_match_date",
        "data_sources",
        "notes",
    ]
    rankings = df[columns].copy()

    return RankingResult(
        rankings=rankings,
        weights_effective=eff,
        cutoff=cutoff,
        matches_total=matches_total,
        matches_in_window=len(matches),
        teams_ranked=len(rankings),
        notes=notes,
        sources=[PRIMARY_SOURCE],
        elo_state=elo_state,
    )


def write_outputs(result: RankingResult, output_dir: str | Path, top_n: int) -> dict[str, Path]:
    """Write top-N and full rankings CSVs. Returns the paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "full_rankings.csv"
    top_path = output_dir / f"top_{top_n}_rankings.csv"
    result.rankings.to_csv(full_path, index=False)
    result.rankings.head(top_n).to_csv(top_path, index=False)
    LOGGER.info("Wrote %s and %s", full_path, top_path)
    return {"full": full_path, "top": top_path}
