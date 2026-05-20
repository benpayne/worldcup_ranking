"""Tests for the Elo module."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from worldcup_ranker.config import EloConfig
from worldcup_ranker.data_sources import filter_before
from worldcup_ranker.elo import (
    compute_elo,
    expected_score,
    goal_difference_multiplier,
    match_result,
)


def test_expected_score_symmetric():
    assert math.isclose(expected_score(1500, 1500), 0.5, abs_tol=1e-9)


def test_expected_score_higher_rating_wins_more():
    high = expected_score(1800, 1500)
    low = expected_score(1500, 1800)
    assert high > 0.8
    assert math.isclose(high + low, 1.0, abs_tol=1e-9)


def test_goal_difference_multiplier():
    assert goal_difference_multiplier(0) == 1.0
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(3) == pytest.approx(14 / 8)
    assert goal_difference_multiplier(-4) == pytest.approx(15 / 8)


def test_match_result():
    assert match_result(2, 1) == (1.0, 0.0)
    assert match_result(0, 3) == (0.0, 1.0)
    assert match_result(1, 1) == (0.5, 0.5)


def test_compute_elo_only_uses_matches_before_cutoff(tiny_matches):
    cfg = EloConfig(
        initial_rating=1500,
        home_advantage=65,
        k_base=30,
        importance={"FIFA World Cup": 4.0, "FIFA World Cup qualification": 2.0},
        default_importance=1.0,
        goal_diff_multiplier=True,
    )
    cutoff = pd.Timestamp("2026-06-11")

    pre = filter_before(tiny_matches, cutoff)
    state_pre = compute_elo(pre, cfg)

    # Calling with the full frame after manually filtering must match.
    state_full_filtered = compute_elo(filter_before(tiny_matches, cutoff), cfg)
    for team, rating in state_pre.ratings.items():
        assert state_full_filtered.ratings[team] == pytest.approx(rating)

    # All teams' last match date is before the cutoff.
    for team, last_date in state_pre.last_match_date.items():
        assert last_date < cutoff, f"{team} last_match_date {last_date} >= cutoff {cutoff}"


def test_compute_elo_post_cutoff_does_not_leak(tiny_matches):
    cfg = EloConfig()
    cutoff = pd.Timestamp("2026-06-11")

    pre = filter_before(tiny_matches, cutoff)
    state_pre = compute_elo(pre, cfg)

    # Sanity: feeding in post-cutoff matches *would* change the ratings.
    state_all = compute_elo(tiny_matches, cfg)
    assert state_pre.ratings["Brazil"] != pytest.approx(state_all.ratings["Brazil"]), (
        "Test fixture is too weak to detect leakage; please strengthen it."
    )


def test_home_advantage_affects_expected_score():
    cfg = EloConfig(home_advantage=100, k_base=30, default_importance=1.0)
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "x",
                "country": "x",
                "neutral": False,
            }
        ]
    )
    state = compute_elo(matches, cfg)
    # Home draw against an equal-rated visitor should slightly lower home Elo
    # (home was expected to win because of the +100 boost).
    assert state.ratings["A"] < cfg.initial_rating
    assert state.ratings["B"] > cfg.initial_rating


def test_neutral_disables_home_advantage():
    cfg = EloConfig(home_advantage=100, k_base=30, default_importance=1.0)
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "x",
                "country": "x",
                "neutral": True,
            }
        ]
    )
    state = compute_elo(matches, cfg)
    # Equal ratings + neutral + draw -> no Elo change.
    assert state.ratings["A"] == pytest.approx(cfg.initial_rating)
    assert state.ratings["B"] == pytest.approx(cfg.initial_rating)


def test_tournament_importance_multiplier_amplifies_change():
    base_cfg = EloConfig(importance={"X": 1.0}, default_importance=1.0)
    big_cfg = EloConfig(importance={"X": 4.0}, default_importance=1.0)
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 2,
                "away_score": 0,
                "tournament": "X",
                "city": "x",
                "country": "x",
                "neutral": True,
            }
        ]
    )
    base = compute_elo(matches, base_cfg)
    big = compute_elo(matches, big_cfg)
    # The bigger-importance run produces a larger rating swing.
    assert (big.ratings["A"] - 1500) > (base.ratings["A"] - 1500)
