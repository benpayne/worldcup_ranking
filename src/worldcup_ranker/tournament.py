"""Group-stage rules + knockout bracket structure.

Group tiebreakers (FIFA Article 25, 2026 cycle):
  1. Total points (W=3, D=1, L=0)
  2. Overall goal difference
  3. Overall goals scored
  4. Head-to-head points among tied teams
  5. Head-to-head goal difference among tied teams
  6. Head-to-head goals scored among tied teams
  7. Fair-play points -> we don't model these; we use a stable random tiebreak
  8. Drawing of lots -> same fallback

Best-eight-third-place teams are ranked by the same tiebreakers using
overall (not head-to-head) records, plus a stable random fallback.

Bracket assignment for the Round of 32 is an intentional simplification
of FIFA's Annex C 495-row lookup table:
  - Rank the 32 advancers by a tournament-seed score derived from their
    group finishing position (1st/2nd/3rd-place) and tiebreaker quality.
  - Pair seeds 1-32 like a single-elimination bracket so the top seed
    plays the worst-qualified 3rd-place team in R32, while keeping
    same-group teams apart until at least the SF.
  - Two-side bracket so the top two seeds can only meet in the final.

See the README for caveats on the simplification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

from .score_model import (
    MatchOutcome,
    ScoreModelParams,
    deterministic_match,
    simulate_match,
)

GROUP_LETTERS = list("ABCDEFGHIJKL")  # 12 groups


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TeamRecord:
    """Per-team standings record within a group."""

    team: str
    group: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    # Head-to-head results against tied opponents (filled when computing standings).
    tiebreak_rng_key: float = 0.0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class GroupMatch:
    """A single group-stage fixture (neutral venue is assumed)."""

    group: str
    team_a: str
    team_b: str
    outcome: Optional[MatchOutcome] = None


@dataclass
class KnockoutMatch:
    """A knockout-stage fixture."""

    round_name: str
    slot_id: int
    team_a: str
    team_b: str
    outcome: Optional[MatchOutcome] = None

    @property
    def winner_team(self) -> str:
        if self.outcome is None or self.outcome.winner is None:
            raise ValueError(f"No winner yet for {self.round_name} slot {self.slot_id}")
        return self.team_a if self.outcome.winner == "A" else self.team_b


@dataclass
class SimulatedTournament:
    """Result bundle returned by one run (deterministic or one MC iteration)."""

    group_matches: list[GroupMatch] = field(default_factory=list)
    group_standings: dict[str, list[TeamRecord]] = field(default_factory=dict)
    advancers: list[TeamRecord] = field(default_factory=list)
    knockout_matches: list[KnockoutMatch] = field(default_factory=list)
    champion: Optional[str] = None
    runner_up: Optional[str] = None
    third_place: Optional[str] = None
    fourth_place: Optional[str] = None


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------


def build_group_matches(groups: dict[str, list[str]]) -> list[GroupMatch]:
    """Round-robin: every pair within a group plays once."""
    matches = []
    for letter, teams in groups.items():
        for a, b in combinations(teams, 2):
            matches.append(GroupMatch(group=letter, team_a=a, team_b=b))
    return matches


def _apply_result(record: TeamRecord, gf: int, ga: int) -> None:
    record.played += 1
    record.goals_for += gf
    record.goals_against += ga
    if gf > ga:
        record.wins += 1
        record.points += 3
    elif gf < ga:
        record.losses += 1
    else:
        record.draws += 1
        record.points += 1


def play_group_stage(
    groups: dict[str, list[str]],
    scores: dict[str, float],
    params: ScoreModelParams,
    rng: np.random.Generator,
    deterministic: bool,
) -> tuple[list[GroupMatch], dict[str, dict[str, TeamRecord]]]:
    """Run every group match; return matches with outcomes and per-group records."""
    matches = build_group_matches(groups)
    records: dict[str, dict[str, TeamRecord]] = {
        letter: {t: TeamRecord(team=t, group=letter) for t in teams}
        for letter, teams in groups.items()
    }

    for match in matches:
        score_a = scores.get(match.team_a, 50.0)
        score_b = scores.get(match.team_b, 50.0)
        if deterministic:
            outcome = deterministic_match(
                score_a, score_b, params, knockout=False, neutral=True
            )
        else:
            outcome = simulate_match(
                score_a, score_b, params, rng, knockout=False, neutral=True
            )
        match.outcome = outcome
        _apply_result(records[match.group][match.team_a], outcome.goals_a, outcome.goals_b)
        _apply_result(records[match.group][match.team_b], outcome.goals_b, outcome.goals_a)

    # Assign a stable per-team random key for the final fallback tiebreaker.
    for letter, recs in records.items():
        for rec in recs.values():
            rec.tiebreak_rng_key = float(rng.random())

    return matches, records


def _head_to_head_record(
    teams_to_compare: list[TeamRecord],
    matches: list[GroupMatch],
) -> dict[str, tuple[int, int, int]]:
    """Return (points, gd, gf) for each team computed only over matches
    among the supplied subset."""
    names = {r.team for r in teams_to_compare}
    h2h: dict[str, dict[str, int]] = {
        r.team: {"pts": 0, "gf": 0, "ga": 0} for r in teams_to_compare
    }
    for m in matches:
        if m.team_a in names and m.team_b in names and m.outcome is not None:
            ga = m.outcome.total_goals_a
            gb = m.outcome.total_goals_b
            h2h[m.team_a]["gf"] += ga
            h2h[m.team_a]["ga"] += gb
            h2h[m.team_b]["gf"] += gb
            h2h[m.team_b]["ga"] += ga
            if ga > gb:
                h2h[m.team_a]["pts"] += 3
            elif ga < gb:
                h2h[m.team_b]["pts"] += 3
            else:
                h2h[m.team_a]["pts"] += 1
                h2h[m.team_b]["pts"] += 1
    return {
        team: (vals["pts"], vals["gf"] - vals["ga"], vals["gf"])
        for team, vals in h2h.items()
    }


def rank_group(
    records: dict[str, TeamRecord], matches: list[GroupMatch]
) -> list[TeamRecord]:
    """Apply FIFA group tiebreakers and return teams sorted best-to-worst."""
    # First pass: overall pts -> GD -> GF.
    ordered = sorted(
        records.values(),
        key=lambda r: (-r.points, -r.goal_difference, -r.goals_for),
    )

    # Resolve ties using head-to-head then random key.
    result: list[TeamRecord] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and (
            ordered[j].points == ordered[i].points
            and ordered[j].goal_difference == ordered[i].goal_difference
            and ordered[j].goals_for == ordered[i].goals_for
        ):
            j += 1
        tied = ordered[i:j]
        if len(tied) == 1:
            result.append(tied[0])
        else:
            h2h = _head_to_head_record(tied, matches)
            tied_sorted = sorted(
                tied,
                key=lambda r: (
                    -h2h[r.team][0],   # h2h points
                    -h2h[r.team][1],   # h2h GD
                    -h2h[r.team][2],   # h2h GF
                    -r.tiebreak_rng_key,
                ),
            )
            result.extend(tied_sorted)
        i = j
    return result


# ---------------------------------------------------------------------------
# Best-third-place selection
# ---------------------------------------------------------------------------


def rank_third_place(
    standings: dict[str, list[TeamRecord]],
) -> list[TeamRecord]:
    """Return all 12 third-place teams sorted by their overall record."""
    thirds = [s[2] for s in standings.values() if len(s) >= 3]
    return sorted(
        thirds,
        key=lambda r: (
            -r.points,
            -r.goal_difference,
            -r.goals_for,
            -r.tiebreak_rng_key,
        ),
    )


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------


def _seed_value(record: TeamRecord, position: int) -> tuple[int, int, int, int]:
    """Return a sortable key: best (lowest) seed first.

    Group winners outrank runners-up; runners-up outrank third-placed; within
    a finishing position, points -> GD -> GF (-tiebreak as final fallback).
    """
    return (
        position,                # 1=winner, 2=runner-up, 3=third
        -record.points,
        -record.goal_difference,
        -record.goals_for,
    )


def select_advancers(
    standings: dict[str, list[TeamRecord]],
) -> list[tuple[TeamRecord, int]]:
    """Return the 32 advancing teams paired with their finishing position."""
    advancers: list[tuple[TeamRecord, int]] = []
    for group in GROUP_LETTERS:
        s = standings[group]
        advancers.append((s[0], 1))
        advancers.append((s[1], 2))
    best_thirds = rank_third_place(standings)[:8]
    for rec in best_thirds:
        advancers.append((rec, 3))
    return advancers


def _bracket_seed_order(n: int) -> list[int]:
    """Standard single-elimination seeding order for n participants.

    For n=32: returns [1, 32, 16, 17, 8, 25, 9, 24, 4, 29, 13, 20, 5, 28, 12,
                      21, 2, 31, 15, 18, 7, 26, 10, 23, 3, 30, 14, 19, 6, 27,
                      11, 22].
    The bracket is built so that seed 1 meets seed 2 only in the final, and
    each pair (i, n+1-i) plays in round 1.
    """
    order = [1, 2]
    while len(order) < n:
        next_order = []
        m = 2 * len(order) + 1
        for s in order:
            next_order.append(s)
            next_order.append(m - s)
        order = next_order
    return order


def _no_same_group(pair: tuple[TeamRecord, TeamRecord]) -> bool:
    return pair[0].group != pair[1].group


def build_round_of_32(
    advancers: list[tuple[TeamRecord, int]],
    rng: np.random.Generator,
) -> list[KnockoutMatch]:
    """Pair the 32 advancers into 16 R32 matches.

    Approximation of FIFA's Annex C: rank all 32 by (finishing position,
    points, GD, GF), then pair according to the standard 32-team bracket
    seeding order, swapping pairs as needed to avoid same-group rematches.
    """
    if len(advancers) != 32:
        raise ValueError(f"Need exactly 32 advancers, got {len(advancers)}")

    ranked = sorted(advancers, key=lambda x: _seed_value(x[0], x[1]))
    seed_to_team: dict[int, TeamRecord] = {
        seed: rec for seed, (rec, _pos) in enumerate(ranked, start=1)
    }

    order = _bracket_seed_order(32)
    pairs: list[tuple[TeamRecord, TeamRecord]] = []
    for i in range(0, 32, 2):
        a = seed_to_team[order[i]]
        b = seed_to_team[order[i + 1]]
        pairs.append((a, b))

    # Resolve same-group conflicts by swapping the offending second slot with
    # another pair's second slot until no conflict remains.
    max_swaps = 100
    for _ in range(max_swaps):
        conflicts = [idx for idx, p in enumerate(pairs) if not _no_same_group(p)]
        if not conflicts:
            break
        # Take the first conflict; find a pair to swap its B-slot with.
        idx = conflicts[0]
        a, b = pairs[idx]
        swap_candidates = [
            k
            for k, (a2, b2) in enumerate(pairs)
            if k != idx
            and a.group != b2.group
            and a2.group != b.group
            and _no_same_group((a2, b2))  # don't break a clean pair
        ]
        if not swap_candidates:
            # Fall back: shuffle the un-paired side and try again.
            rng.shuffle(swap_candidates)
            break
        k = swap_candidates[0]
        pairs[idx] = (a, pairs[k][1])
        pairs[k] = (pairs[k][0], b)

    matches = []
    for slot, (a, b) in enumerate(pairs, start=1):
        matches.append(
            KnockoutMatch(round_name="R32", slot_id=slot, team_a=a.team, team_b=b.team)
        )
    return matches


def _pair_winners(round_name: str, prev: list[KnockoutMatch]) -> list[KnockoutMatch]:
    """Pair consecutive winners from ``prev`` into the next round."""
    next_round = []
    for i in range(0, len(prev), 2):
        next_round.append(
            KnockoutMatch(
                round_name=round_name,
                slot_id=(i // 2) + 1,
                team_a=prev[i].winner_team,
                team_b=prev[i + 1].winner_team,
            )
        )
    return next_round


# ---------------------------------------------------------------------------
# Knockout play-through
# ---------------------------------------------------------------------------


def _play_knockout_round(
    matches: list[KnockoutMatch],
    scores: dict[str, float],
    params: ScoreModelParams,
    rng: np.random.Generator,
    deterministic: bool,
) -> None:
    for m in matches:
        sa = scores.get(m.team_a, 50.0)
        sb = scores.get(m.team_b, 50.0)
        if deterministic:
            m.outcome = deterministic_match(sa, sb, params, knockout=True, neutral=True)
        else:
            m.outcome = simulate_match(sa, sb, params, rng, knockout=True, neutral=True)


def play_knockouts(
    r32: list[KnockoutMatch],
    scores: dict[str, float],
    params: ScoreModelParams,
    rng: np.random.Generator,
    deterministic: bool,
) -> tuple[list[KnockoutMatch], str, str, str, str]:
    """Run R32 -> R16 -> QF -> SF -> 3rd-place + Final. Return the rounds + medallists."""
    _play_knockout_round(r32, scores, params, rng, deterministic)
    r16 = _pair_winners("R16", r32)
    _play_knockout_round(r16, scores, params, rng, deterministic)
    qf = _pair_winners("QF", r16)
    _play_knockout_round(qf, scores, params, rng, deterministic)
    sf = _pair_winners("SF", qf)
    _play_knockout_round(sf, scores, params, rng, deterministic)

    sf_losers = [
        (m.team_b if m.outcome.winner == "A" else m.team_a)
        for m in sf
    ]
    third_place_match = KnockoutMatch(
        round_name="3rd-place",
        slot_id=1,
        team_a=sf_losers[0],
        team_b=sf_losers[1],
    )
    final_match = KnockoutMatch(
        round_name="Final",
        slot_id=1,
        team_a=sf[0].winner_team,
        team_b=sf[1].winner_team,
    )
    _play_knockout_round([third_place_match, final_match], scores, params, rng, deterministic)

    champion = final_match.winner_team
    runner_up = final_match.team_b if final_match.outcome.winner == "A" else final_match.team_a
    third = third_place_match.winner_team
    fourth = third_place_match.team_b if third_place_match.outcome.winner == "A" else third_place_match.team_a

    all_matches = r32 + r16 + qf + sf + [third_place_match, final_match]
    return all_matches, champion, runner_up, third, fourth


# ---------------------------------------------------------------------------
# Top-level: play one tournament
# ---------------------------------------------------------------------------


def play_tournament(
    groups: dict[str, list[str]],
    scores: dict[str, float],
    params: ScoreModelParams,
    rng: np.random.Generator,
    deterministic: bool,
) -> SimulatedTournament:
    """One full simulation (group stage -> bracket -> medals)."""
    matches, records = play_group_stage(groups, scores, params, rng, deterministic)
    standings = {
        letter: rank_group(records[letter], matches) for letter in records
    }
    advancers_with_pos = select_advancers(standings)
    advancers_records = [rec for rec, _ in advancers_with_pos]

    r32 = build_round_of_32(advancers_with_pos, rng)
    knockouts, champion, runner_up, third, fourth = play_knockouts(
        r32, scores, params, rng, deterministic
    )

    return SimulatedTournament(
        group_matches=matches,
        group_standings=standings,
        advancers=advancers_records,
        knockout_matches=knockouts,
        champion=champion,
        runner_up=runner_up,
        third_place=third,
        fourth_place=fourth,
    )
