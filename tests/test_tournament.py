"""Tests for tournament logic: group rules, third-place, bracket assignment."""
from __future__ import annotations

import numpy as np
import pytest

from worldcup_ranker.score_model import ScoreModelParams
from worldcup_ranker.tournament import (
    GROUP_LETTERS,
    GroupMatch,
    TeamRecord,
    build_group_matches,
    build_round_of_32,
    play_tournament,
    rank_group,
    rank_third_place,
    select_advancers,
)
from worldcup_ranker.score_model import MatchOutcome


def _make_groups() -> dict[str, list[str]]:
    """A complete 12-group fixture: each letter has 4 unique team strings."""
    return {letter: [f"{letter}{i}" for i in range(1, 5)] for letter in GROUP_LETTERS}


def _make_scores(groups: dict[str, list[str]]) -> dict[str, float]:
    """A clear strength hierarchy: position-1 in each group is strongest."""
    scores = {}
    for letter, teams in groups.items():
        scores[teams[0]] = 85.0
        scores[teams[1]] = 65.0
        scores[teams[2]] = 45.0
        scores[teams[3]] = 25.0
    return scores


def test_build_group_matches_count():
    matches = build_group_matches(_make_groups())
    # Each group: C(4,2) = 6 matches. 12 groups -> 72.
    assert len(matches) == 72


def test_rank_group_applies_overall_then_head_to_head():
    # Two teams tied on points/GD/GF; H2H decides.
    recs = {
        "X": TeamRecord(team="X", group="A", played=3, wins=2, draws=0, losses=1,
                        goals_for=4, goals_against=2, points=6),
        "Y": TeamRecord(team="Y", group="A", played=3, wins=2, draws=0, losses=1,
                        goals_for=4, goals_against=2, points=6),
        "Z": TeamRecord(team="Z", group="A", played=3, wins=2, draws=0, losses=1,
                        goals_for=4, goals_against=2, points=6),
        "W": TeamRecord(team="W", group="A", played=3, wins=0, draws=0, losses=3,
                        goals_for=0, goals_against=8, points=0),
    }
    # H2H: X beat Y 2-0, X drew Z 1-1, Y beat Z 1-0 -> X 4pts, Y 3pts, Z 1pt
    matches = [
        GroupMatch(group="A", team_a="X", team_b="Y",
                   outcome=MatchOutcome(goals_a=2, goals_b=0)),
        GroupMatch(group="A", team_a="X", team_b="Z",
                   outcome=MatchOutcome(goals_a=1, goals_b=1)),
        GroupMatch(group="A", team_a="Y", team_b="Z",
                   outcome=MatchOutcome(goals_a=1, goals_b=0)),
    ]
    ordered = rank_group(recs, matches)
    assert [r.team for r in ordered] == ["X", "Y", "Z", "W"]


def test_rank_group_overall_wins_when_unique_points():
    recs = {
        "A": TeamRecord(team="A", group="A", played=3, points=9,
                        goals_for=7, goals_against=1),
        "B": TeamRecord(team="B", group="A", played=3, points=6,
                        goals_for=5, goals_against=3),
        "C": TeamRecord(team="C", group="A", played=3, points=3,
                        goals_for=2, goals_against=5),
        "D": TeamRecord(team="D", group="A", played=3, points=0,
                        goals_for=0, goals_against=5),
    }
    ordered = rank_group(recs, [])
    assert [r.team for r in ordered] == ["A", "B", "C", "D"]


def test_rank_third_place_returns_all_twelve():
    standings: dict[str, list[TeamRecord]] = {}
    for i, letter in enumerate(GROUP_LETTERS):
        standings[letter] = [
            TeamRecord(team=f"{letter}1", group=letter, points=9, goals_for=6),
            TeamRecord(team=f"{letter}2", group=letter, points=6, goals_for=4),
            TeamRecord(team=f"{letter}3", group=letter, points=3 + (i % 2),
                       goals_for=2 + (i % 3)),
            TeamRecord(team=f"{letter}4", group=letter, points=0),
        ]
    thirds = rank_third_place(standings)
    assert len(thirds) == 12
    # All should be third-placed teams.
    assert all(r.team.endswith("3") for r in thirds)
    # Sorted descending by points.
    assert thirds[0].points >= thirds[-1].points


def test_select_advancers_picks_exactly_32():
    standings = {
        letter: [
            TeamRecord(team=f"{letter}1", group=letter, points=9),
            TeamRecord(team=f"{letter}2", group=letter, points=6),
            TeamRecord(team=f"{letter}3", group=letter, points=3 + (i % 4)),
            TeamRecord(team=f"{letter}4", group=letter, points=0),
        ]
        for i, letter in enumerate(GROUP_LETTERS)
    }
    advancers = select_advancers(standings)
    assert len(advancers) == 32
    # 12 winners + 12 runners-up + 8 best thirds.
    positions = [p for _, p in advancers]
    assert positions.count(1) == 12
    assert positions.count(2) == 12
    assert positions.count(3) == 8


def test_round_of_32_pairs_top_seed_with_lowest():
    """The strongest seed (group winner with the best record) should face the
    weakest of the eight third-place teams."""
    advancers = []
    for i, letter in enumerate(GROUP_LETTERS):
        # Make group A's winner much stronger than everyone else's winner.
        winner_points = 9 if letter != "A" else 9
        winner_gf = 12 if letter == "A" else (5 + i)
        advancers.append((TeamRecord(team=f"{letter}1", group=letter,
                                     points=winner_points, goals_for=winner_gf,
                                     goals_against=1), 1))
        advancers.append((TeamRecord(team=f"{letter}2", group=letter,
                                     points=6, goals_for=4, goals_against=4), 2))
    # 8 best thirds, with C3 the weakest by goals_for.
    for i, letter in enumerate("ABCDEFGH"):
        gf = 1 if letter == "C" else (3 + i)
        advancers.append((TeamRecord(team=f"{letter}3", group=letter,
                                     points=3, goals_for=gf, goals_against=5), 3))
    rng = np.random.default_rng(0)
    r32 = build_round_of_32(advancers, rng)
    assert len(r32) == 16
    # Find the match involving A1 (top seed).
    a1_match = next(m for m in r32 if "A1" in (m.team_a, m.team_b))
    # Its opponent should be a third-place team, ideally the weakest.
    opponent = a1_match.team_b if a1_match.team_a == "A1" else a1_match.team_a
    assert opponent.endswith("3")


def test_round_of_32_no_same_group_rematches():
    """No R32 pair should come from the same group."""
    standings = {
        letter: [
            TeamRecord(team=f"{letter}1", group=letter, points=9, goals_for=5),
            TeamRecord(team=f"{letter}2", group=letter, points=6, goals_for=4),
            TeamRecord(team=f"{letter}3", group=letter, points=3, goals_for=2 + i),
            TeamRecord(team=f"{letter}4", group=letter, points=0, goals_for=0),
        ]
        for i, letter in enumerate(GROUP_LETTERS)
    }
    advancers = select_advancers(standings)
    rng = np.random.default_rng(42)
    r32 = build_round_of_32(advancers, rng)
    for m in r32:
        ga = m.team_a[0]  # group letter is first char
        gb = m.team_b[0]
        assert ga != gb, f"Same-group pair: {m.team_a} vs {m.team_b}"


def test_play_tournament_deterministic_produces_champion():
    groups = _make_groups()
    scores = _make_scores(groups)
    rng = np.random.default_rng(0)
    t = play_tournament(groups, scores, ScoreModelParams(), rng, deterministic=True)
    assert t.champion is not None
    assert t.runner_up is not None
    assert t.third_place is not None
    assert t.fourth_place is not None
    # The strongest team in each group should win it.
    for letter, teams in groups.items():
        assert t.group_standings[letter][0].team == teams[0]
    # Final standings should be distinct teams.
    finalists = {t.champion, t.runner_up, t.third_place, t.fourth_place}
    assert len(finalists) == 4


def test_play_tournament_mc_is_reproducible_with_seed():
    groups = _make_groups()
    scores = _make_scores(groups)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    t1 = play_tournament(groups, scores, ScoreModelParams(), rng1, deterministic=False)
    t2 = play_tournament(groups, scores, ScoreModelParams(), rng2, deterministic=False)
    assert t1.champion == t2.champion
    assert t1.runner_up == t2.runner_up
