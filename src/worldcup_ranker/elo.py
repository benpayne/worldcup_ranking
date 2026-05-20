"""Elo rating computation for national teams.

The implementation roughly mirrors the World Football Elo Ratings model
(eloratings.net):

    expected   = 1 / (1 + 10 ** (-rating_diff / 400))
    new_rating = old_rating + K * G * I * (result - expected)

where ``G`` is a goal-difference multiplier and ``I`` is a match-importance
multiplier configured per tournament. Home advantage is added to the home
team's rating when computing the expected score (unless the match is at a
neutral venue).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .config import EloConfig


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B given the two ratings (no draw modelling)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def goal_difference_multiplier(goal_diff: int) -> float:
    """Davidson-style multiplier used by World Football Elo Ratings."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    # 11/8 + N/8 for g >= 3
    return (11.0 + g) / 8.0


def match_result(home_score: int, away_score: int) -> tuple[float, float]:
    """Return (home_result, away_result) where 1.0 = win, 0.5 = draw, 0 = loss."""
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


@dataclass
class EloState:
    """Mutable container for Elo ratings keyed by team name."""

    initial_rating: float = 1500.0
    ratings: dict[str, float] = field(default_factory=dict)
    last_match_date: dict[str, pd.Timestamp] = field(default_factory=dict)
    matches_played: dict[str, int] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.setdefault(team, self.initial_rating)

    def set(self, team: str, value: float, date: pd.Timestamp) -> None:
        self.ratings[team] = value
        self.last_match_date[team] = date
        self.matches_played[team] = self.matches_played.get(team, 0) + 1

    def as_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "team": team,
                "elo": rating,
                "last_match_date": self.last_match_date.get(team),
                "matches_played": self.matches_played.get(team, 0),
            }
            for team, rating in self.ratings.items()
        ]
        return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def compute_elo(matches: pd.DataFrame, config: EloConfig) -> EloState:
    """Run the Elo algorithm over a chronologically sorted match dataframe.

    Parameters
    ----------
    matches:
        DataFrame with columns ``date``, ``home_team``, ``away_team``,
        ``home_score``, ``away_score``, ``tournament``, ``neutral``. Must be
        sorted ascending by ``date``; date filtering must already have been
        applied so that no post-cutoff matches leak in.
    config:
        Validated EloConfig.
    """
    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date").reset_index(drop=True)

    state = EloState(initial_rating=config.initial_rating)

    home_teams = matches["home_team"].to_numpy()
    away_teams = matches["away_team"].to_numpy()
    home_scores = matches["home_score"].to_numpy()
    away_scores = matches["away_score"].to_numpy()
    tournaments = matches["tournament"].to_numpy()
    neutrals = matches["neutral"].to_numpy()
    dates = matches["date"].to_numpy()

    for i in range(len(matches)):
        home = home_teams[i]
        away = away_teams[i]
        hs = int(home_scores[i])
        as_ = int(away_scores[i])
        tournament = tournaments[i]
        neutral = bool(neutrals[i])
        date = pd.Timestamp(dates[i])

        r_home = state.get(home)
        r_away = state.get(away)
        eff_home = r_home + (0.0 if neutral else config.home_advantage)

        expected_home = expected_score(eff_home, r_away)
        expected_away = 1.0 - expected_home

        result_home, result_away = match_result(hs, as_)

        importance = config.importance.get(tournament, config.default_importance)
        g_mult = goal_difference_multiplier(hs - as_) if config.goal_diff_multiplier else 1.0
        k_eff = config.k_base * importance * g_mult

        new_home = r_home + k_eff * (result_home - expected_home)
        new_away = r_away + k_eff * (result_away - expected_away)

        state.set(home, new_home, date)
        state.set(away, new_away, date)

    return state


def elo_for_teams(state: EloState, teams: Iterable[str]) -> pd.DataFrame:
    """Return a DataFrame of Elo ratings for the requested teams.

    Missing teams (no matches in the input window) are returned with the
    initial rating and zero matches; this keeps the ranking pipeline robust
    to typos / minor name mismatches.
    """
    rows = []
    for t in teams:
        rows.append(
            {
                "team": t,
                "elo": state.ratings.get(t, state.initial_rating),
                "last_match_date": state.last_match_date.get(t),
                "matches_played": state.matches_played.get(t, 0),
            }
        )
    return pd.DataFrame(rows)
