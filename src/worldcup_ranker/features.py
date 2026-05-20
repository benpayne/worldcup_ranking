"""Feature computation: recent form, goal performance, squad strength."""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import (
    GoalPerformanceConfig,
    RecentFormConfig,
    SquadStrengthConfig,
)
from .elo import EloState, expected_score


def _per_team_match_view(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two rows (one per team) for easier aggregation."""
    home = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "goals_for": matches["home_score"],
            "goals_against": matches["away_score"],
            "is_home": ~matches["neutral"].astype(bool),
            "is_neutral": matches["neutral"].astype(bool),
            "tournament": matches["tournament"],
        }
    )
    away = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "goals_for": matches["away_score"],
            "goals_against": matches["home_score"],
            "is_home": False,
            "is_neutral": matches["neutral"].astype(bool),
            "tournament": matches["tournament"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def compute_recent_form(
    matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_days: int,
    config: RecentFormConfig,
    elo_state: Optional[EloState] = None,
) -> pd.DataFrame:
    """Compute a per-team recent-form index using exponentially decayed results.

    Each match contributes ``result_value - expected_value`` (i.e. performance
    relative to expectation), weighted by ``0.5 ** (age / half_life)``. The
    final ``recent_form_raw`` column is the weighted mean of those signed
    contributions; this gives a value roughly in [-1, 1] which is rescaled to
    [0, 100] downstream.

    ``expected_value`` is derived from the pre-match Elo state when
    ``opponent_adjusted=True``; otherwise it is fixed at 0.5.
    """
    window_start = cutoff - pd.Timedelta(days=window_days)
    window = matches.loc[
        (matches["date"] >= window_start) & (matches["date"] < cutoff)
    ].copy()

    if window.empty:
        return pd.DataFrame(columns=["team", "recent_form_raw", "matches_used", "last_match_date"])

    view = _per_team_match_view(window)
    view["result"] = np.where(
        view["goals_for"] > view["goals_against"],
        1.0,
        np.where(view["goals_for"] < view["goals_against"], 0.0, 0.5),
    )

    age_days = (cutoff - view["date"]).dt.total_seconds() / 86400.0
    half_life = max(config.half_life_days, 1.0)
    view["weight"] = np.power(0.5, age_days / half_life)

    if config.opponent_adjusted and elo_state is not None:
        team_elo = view["team"].map(lambda t: elo_state.ratings.get(t, elo_state.initial_rating))
        opp_elo = view["opponent"].map(lambda t: elo_state.ratings.get(t, elo_state.initial_rating))
        view["expected"] = [
            expected_score(float(a), float(b)) for a, b in zip(team_elo, opp_elo)
        ]
    else:
        view["expected"] = 0.5

    view["signed_contrib"] = (view["result"] - view["expected"]) * view["weight"]

    agg = view.groupby("team").agg(
        weight_sum=("weight", "sum"),
        signed_sum=("signed_contrib", "sum"),
        matches_used=("date", "count"),
        last_match_date=("date", "max"),
    )
    agg["recent_form_raw"] = agg["signed_sum"] / agg["weight_sum"].replace(0, np.nan)
    agg = agg.reset_index()
    return agg[["team", "recent_form_raw", "matches_used", "last_match_date"]]


def compute_goal_performance(
    matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_days: int,
    config: GoalPerformanceConfig,
    xg_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute an opponent-naïve goal-based performance index.

    When ``xg_df`` is provided (columns: ``date, home_team, away_team,
    home_xg, away_xg``), per-match xG/xGA replace the capped goals proxy
    for matches present in the CSV. Matches not in the CSV fall through
    to capped goals. Mixing the two scales is intentional: xG and goals
    are on similar scales (typical match ~2 each), so an averaging blend
    is a reasonable enrichment.

    For each team:
        attack  = mean(effective_for)
        defense = mean(effective_against)  (inverted later)
        gd      = mean(clip(effective_for - effective_against, -cap, +cap))

    The raw index is ``attack_weight * attack - defense_weight * defense``,
    which downstream is rescaled to 0-100.
    """
    window_start = cutoff - pd.Timedelta(days=window_days)
    window = matches.loc[
        (matches["date"] >= window_start) & (matches["date"] < cutoff)
    ].copy()

    empty_cols = [
        "team",
        "goal_perf_raw",
        "goals_for_avg",
        "goals_against_avg",
        "goal_diff_avg",
        "xg_matches_used",
    ]
    if window.empty:
        return pd.DataFrame(columns=empty_cols)

    view = _per_team_match_view(window)
    cap = max(config.goal_diff_cap, 0)
    view["goals_for_capped"] = (
        view["goals_for"].clip(upper=cap) if cap > 0 else view["goals_for"]
    )
    view["goals_against_capped"] = (
        view["goals_against"].clip(upper=cap) if cap > 0 else view["goals_against"]
    )

    # Optionally merge in xG.
    if xg_df is not None and not xg_df.empty:
        xg_long = _xg_long_view(xg_df)
        view = view.merge(
            xg_long, on=["date", "team", "opponent"], how="left"
        )
    else:
        view["xg_for"] = np.nan
        view["xg_against"] = np.nan

    view["effective_for"] = view["xg_for"].fillna(view["goals_for_capped"])
    view["effective_against"] = view["xg_against"].fillna(view["goals_against_capped"])
    view["effective_gd"] = (view["effective_for"] - view["effective_against"]).clip(
        lower=-cap if cap > 0 else None,
        upper=cap if cap > 0 else None,
    )

    agg = view.groupby("team").agg(
        goals_for_avg=("effective_for", "mean"),
        goals_against_avg=("effective_against", "mean"),
        goal_diff_avg=("effective_gd", "mean"),
        xg_matches_used=("xg_for", lambda s: int(s.notna().sum())),
    )
    agg["goal_perf_raw"] = (
        config.attack_weight * agg["goals_for_avg"]
        - config.defense_weight * agg["goals_against_avg"]
    )
    return agg.reset_index()


def _xg_long_view(xg_df: pd.DataFrame) -> pd.DataFrame:
    """Explode an xG match-frame into a team-perspective view.

    Columns: date, team, opponent, xg_for, xg_against.
    """
    home = pd.DataFrame(
        {
            "date": xg_df["date"],
            "team": xg_df["home_team"],
            "opponent": xg_df["away_team"],
            "xg_for": xg_df["home_xg"].astype(float),
            "xg_against": xg_df["away_xg"].astype(float),
        }
    )
    away = pd.DataFrame(
        {
            "date": xg_df["date"],
            "team": xg_df["away_team"],
            "opponent": xg_df["home_team"],
            "xg_for": xg_df["away_xg"].astype(float),
            "xg_against": xg_df["home_xg"].astype(float),
        }
    )
    return pd.concat([home, away], ignore_index=True)


def compute_squad_strength(
    squad_df: Optional[pd.DataFrame],
    teams: Iterable[str],
    config: SquadStrengthConfig,  # noqa: ARG001 (reserved for future use)
) -> Optional[pd.DataFrame]:
    """Return a per-team squad-strength raw score, or ``None`` if unavailable."""
    if squad_df is None or squad_df.empty:
        return None
    teams_list = list(teams)
    df = squad_df[squad_df["team"].isin(teams_list)].copy()
    if df.empty:
        return None
    df = df.rename(columns={"score": "squad_raw"})
    return df[["team", "squad_raw"]].reset_index(drop=True)


def minmax_rescale(values: pd.Series, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    """Min-max rescale a series. Returns 50.0 for every row when range is zero."""
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or math.isclose(vmin, vmax):
        return pd.Series(np.full(len(values), (hi + lo) / 2.0), index=values.index)
    scaled = (values - vmin) / (vmax - vmin)
    return scaled * (hi - lo) + lo


def invert_score(values: pd.Series) -> pd.Series:
    """Helper: invert a 0-100 score so smaller-is-better becomes bigger-is-better."""
    return 100.0 - values
