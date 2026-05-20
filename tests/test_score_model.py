"""Tests for the score model."""
from __future__ import annotations

import math

import numpy as np
import pytest

from worldcup_ranker.score_model import (
    MatchOutcome,
    ScoreModelParams,
    compute_lambdas,
    deterministic_match,
    deterministic_penalties,
    deterministic_regulation,
    sample_extra_time,
    sample_penalties,
    sample_regulation,
    simulate_match,
)


PARAMS = ScoreModelParams()


def test_lambdas_symmetric_when_scores_equal():
    la, lb = compute_lambdas(50, 50, PARAMS)
    assert math.isclose(la, lb)


def test_lambdas_higher_for_better_team():
    la, lb = compute_lambdas(80, 40, PARAMS)
    assert la > lb


def test_lambdas_home_advantage_increases_home_rate():
    la_neutral, lb_neutral = compute_lambdas(50, 50, PARAMS, neutral=True)
    la_home, lb_home = compute_lambdas(50, 50, PARAMS, a_is_home=True, neutral=False)
    assert la_home > la_neutral
    assert lb_home < lb_neutral


def test_lambdas_minimum_clamp():
    la, _ = compute_lambdas(0, 100, ScoreModelParams(min_lambda=0.2))
    assert la >= 0.2


def test_deterministic_regulation_breaks_ties_toward_favourite():
    """A clearly stronger team should not deterministically draw."""
    ga, gb = deterministic_regulation(85, 50, PARAMS)
    assert ga > gb


def test_deterministic_regulation_can_be_equal_for_equal_teams():
    ga, gb = deterministic_regulation(50, 50, PARAMS)
    assert ga == gb


def test_sample_regulation_returns_nonneg_ints():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ga, gb = sample_regulation(60, 50, PARAMS, rng)
        assert isinstance(ga, int) and ga >= 0
        assert isinstance(gb, int) and gb >= 0


def test_sample_extra_time_yields_lower_mean_than_regulation():
    rng = np.random.default_rng(0)
    reg = np.mean([sum(sample_regulation(60, 60, PARAMS, rng)) for _ in range(2000)])
    rng = np.random.default_rng(0)
    et = np.mean([sum(sample_extra_time(60, 60, PARAMS, rng)) for _ in range(2000)])
    # ET window is 30 min vs 90 min; expect roughly a third the goals.
    assert et < reg * 0.6


def test_penalty_shootout_favours_stronger_team_slightly():
    rng = np.random.default_rng(0)
    wins_for_a = sum(sample_penalties(90, 30, PARAMS, rng) == "A" for _ in range(5000))
    p_a = wins_for_a / 5000
    # Favoured slightly but not overwhelmingly (cap around 0.5 + pk_skill).
    assert 0.5 < p_a < 0.5 + PARAMS.pk_skill + 0.02


def test_penalty_shootout_50_50_for_equal_teams():
    rng = np.random.default_rng(0)
    wins_for_a = sum(sample_penalties(50, 50, PARAMS, rng) == "A" for _ in range(5000))
    p_a = wins_for_a / 5000
    assert abs(p_a - 0.5) < 0.05


def test_simulate_match_knockout_cannot_draw():
    rng = np.random.default_rng(0)
    for _ in range(50):
        out = simulate_match(50, 50, PARAMS, rng, knockout=True)
        assert out.winner in {"A", "B"}, "Knockout match must have a winner."


def test_simulate_match_group_can_draw():
    rng = np.random.default_rng(0)
    draws = 0
    for _ in range(200):
        out = simulate_match(50, 50, PARAMS, rng, knockout=False)
        if out.winner is None:
            draws += 1
    # Roughly even teams in group play should draw occasionally.
    assert draws > 10


def test_deterministic_match_knockout_picks_a_winner():
    out = deterministic_match(80, 40, PARAMS, knockout=True)
    assert out.winner == "A"


def test_deterministic_penalties_picks_favourite():
    assert deterministic_penalties(80, 40, PARAMS) == "A"
    assert deterministic_penalties(40, 80, PARAMS) == "B"
    assert deterministic_penalties(50, 50, PARAMS) == "A"  # break toward A on exact tie
