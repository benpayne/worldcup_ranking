"""Tests for the features module (recent form, goal performance, rescaling)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_ranker.config import (
    GoalPerformanceConfig,
    RecentFormConfig,
    SquadStrengthConfig,
)
from worldcup_ranker.features import (
    compute_goal_performance,
    compute_recent_form,
    compute_squad_strength,
    invert_score,
    minmax_rescale,
)


def test_minmax_rescale_basic():
    s = pd.Series([0.0, 5.0, 10.0])
    out = minmax_rescale(s)
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[1] == pytest.approx(50.0)
    assert out.iloc[2] == pytest.approx(100.0)


def test_minmax_rescale_constant_returns_midpoint():
    s = pd.Series([3.0, 3.0, 3.0])
    out = minmax_rescale(s)
    assert (out == 50.0).all()


def test_invert_score():
    s = pd.Series([0.0, 25.0, 100.0])
    out = invert_score(s)
    assert list(out) == [100.0, 75.0, 0.0]


def test_recent_form_window_filter(tiny_matches):
    cutoff = pd.Timestamp("2026-06-11")
    out = compute_recent_form(
        tiny_matches,
        cutoff=cutoff,
        window_days=365,  # only last 12 months
        config=RecentFormConfig(opponent_adjusted=False, half_life_days=180),
    )
    # Brazil-Argentina 2025-01-15 should be excluded (>365 days before cutoff).
    # Brazil-Chile 2025-06-05 is also outside the window (cutoff - 365 = 2025-06-11).
    # Brazil should still appear because of 2025-09, 2026-01, 2026-03 matches.
    assert set(out["team"]) == {"Brazil", "Argentina", "Uruguay"}
    brazil = out.set_index("team").loc["Brazil"]
    assert brazil["last_match_date"] < cutoff
    assert brazil["matches_used"] >= 3


def test_recent_form_excludes_post_cutoff(tiny_matches):
    cutoff = pd.Timestamp("2026-06-11")
    out = compute_recent_form(
        tiny_matches,
        cutoff=cutoff,
        window_days=10_000,  # arbitrarily large
        config=RecentFormConfig(opponent_adjusted=False),
    )
    # No team's last match should be on/after the cutoff.
    assert (out["last_match_date"] < cutoff).all()
    # Specifically Argentina's 5-0 win on 2026-07-01 must not be counted.
    arg = out.set_index("team").loc["Argentina"]
    assert arg["last_match_date"] <= pd.Timestamp("2026-01-15")


def test_recent_form_decay_weights_recent_more():
    rows = [
        # old loss
        {
            "date": pd.Timestamp("2024-01-01"),
            "home_team": "A",
            "away_team": "B",
            "home_score": 0,
            "away_score": 3,
            "tournament": "Friendly",
            "city": "x",
            "country": "x",
            "neutral": True,
        },
        # recent win
        {
            "date": pd.Timestamp("2026-05-01"),
            "home_team": "A",
            "away_team": "B",
            "home_score": 3,
            "away_score": 0,
            "tournament": "Friendly",
            "city": "x",
            "country": "x",
            "neutral": True,
        },
    ]
    matches = pd.DataFrame(rows)
    cutoff = pd.Timestamp("2026-06-11")
    out = compute_recent_form(
        matches,
        cutoff=cutoff,
        window_days=10_000,
        config=RecentFormConfig(opponent_adjusted=False, half_life_days=180),
    )
    # A's weighted form should be positive since the recent win dominates.
    a = out.set_index("team").loc["A"]
    assert a["recent_form_raw"] > 0.0


def test_goal_performance_caps_blowouts():
    rows = [
        # massive blowout
        {
            "date": pd.Timestamp("2026-01-01"),
            "home_team": "A",
            "away_team": "B",
            "home_score": 10,
            "away_score": 0,
            "tournament": "Friendly",
            "city": "x",
            "country": "x",
            "neutral": True,
        },
    ]
    df = pd.DataFrame(rows)
    out = compute_goal_performance(
        df,
        cutoff=pd.Timestamp("2026-06-11"),
        window_days=1000,
        config=GoalPerformanceConfig(goal_diff_cap=3, attack_weight=0.5, defense_weight=0.5),
    )
    a = out.set_index("team").loc["A"]
    # 10 goals scored is capped to 3 -> attack_weight * 3 - 0 = 1.5
    assert a["goal_diff_avg"] == pytest.approx(3.0)
    assert a["goals_for_avg"] == pytest.approx(3.0)
    assert a["goal_perf_raw"] == pytest.approx(0.5 * 3.0 - 0.5 * 0.0)


def test_compute_squad_strength_returns_none_when_missing():
    out = compute_squad_strength(None, ["A", "B"], SquadStrengthConfig())
    assert out is None


def test_compute_squad_strength_filters_to_teams():
    df = pd.DataFrame({"team": ["A", "B", "C"], "score": [10.0, 20.0, 30.0]})
    out = compute_squad_strength(df, ["A", "B"], SquadStrengthConfig())
    assert set(out["team"]) == {"A", "B"}
    assert set(out.columns) == {"team", "squad_raw"}
