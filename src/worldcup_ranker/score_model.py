"""Per-match score model used by the tournament simulator.

We use an independent Poisson on goals for each side, with the
expected-goals rate driven by the ranker's final_score (0-100) for each
team. Independent Poisson is the industry baseline for football match
prediction; Dixon-Coles low-score correlation correction is intentionally
omitted for simplicity (the README notes this).

Knockout-stage logic on top:
- Extra time uses the same lambdas scaled by 30/90 (a third the rate over
  a third the time).
- Penalty shootouts in elite men's football are weakly predictable; the
  favourite gets a small skill nudge over a 50/50 coin flip and that's
  it.

All public functions accept an optional numpy Generator for reproducible
Monte Carlo runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

WinnerSide = Literal["A", "B"]


@dataclass(frozen=True)
class ScoreModelParams:
    """Hand-tuned defaults; can be overridden via simulate-config.

    Calibration is a follow-up; these are reasonable values for men's
    international football where mean total goals per match is ~2.5 and
    the strongest sides tend to score roughly 2 more than the weakest.
    """

    base_total: float = 2.5
    gap_coef: float = 2.0
    home_boost: float = 1.20
    away_penalty: float = 0.90
    et_rate_factor: float = 30.0 / 90.0
    pk_skill: float = 0.05  # max nudge above 50% for the favoured side
    min_lambda: float = 0.10


@dataclass
class MatchOutcome:
    """A single simulated (or deterministic) match outcome."""

    goals_a: int
    goals_b: int
    et_goals_a: int = 0
    et_goals_b: int = 0
    went_to_et: bool = False
    went_to_pk: bool = False
    pk_winner: Optional[WinnerSide] = None

    @property
    def total_goals_a(self) -> int:
        return self.goals_a + self.et_goals_a

    @property
    def total_goals_b(self) -> int:
        return self.goals_b + self.et_goals_b

    @property
    def winner(self) -> Optional[WinnerSide]:
        """Returns 'A', 'B', or None if a draw (only possible in groups)."""
        if self.went_to_pk and self.pk_winner is not None:
            return self.pk_winner
        if self.total_goals_a > self.total_goals_b:
            return "A"
        if self.total_goals_b > self.total_goals_a:
            return "B"
        return None


def compute_lambdas(
    score_a: float,
    score_b: float,
    params: ScoreModelParams,
    a_is_home: bool = False,
    neutral: bool = True,
) -> tuple[float, float]:
    """Return expected goals (lambda_a, lambda_b) for a regulation 90 minutes."""
    norm_a = score_a / 100.0
    norm_b = score_b / 100.0
    gap = norm_a - norm_b  # in [-1, 1]
    expected_gd = params.gap_coef * gap

    lambda_a = max(params.min_lambda, (params.base_total + expected_gd) / 2.0)
    lambda_b = max(params.min_lambda, (params.base_total - expected_gd) / 2.0)

    if not neutral:
        if a_is_home:
            lambda_a *= params.home_boost
            lambda_b *= params.away_penalty
        else:
            lambda_b *= params.home_boost
            lambda_a *= params.away_penalty

    return lambda_a, lambda_b


def sample_regulation(
    score_a: float,
    score_b: float,
    params: ScoreModelParams,
    rng: np.random.Generator,
    a_is_home: bool = False,
    neutral: bool = True,
) -> tuple[int, int]:
    """Sample regulation-time goals (Poisson)."""
    la, lb = compute_lambdas(score_a, score_b, params, a_is_home, neutral)
    return int(rng.poisson(la)), int(rng.poisson(lb))


def deterministic_regulation(
    score_a: float,
    score_b: float,
    params: ScoreModelParams,
    a_is_home: bool = False,
    neutral: bool = True,
) -> tuple[int, int]:
    """Deterministic regulation score = rounded expected goals.

    Ties on the expected score break toward the favourite by adding 1
    to whichever side has the higher lambda. Avoids spurious 1-1 / 2-2
    deterministic draws for matchups where one team is clearly stronger.
    """
    la, lb = compute_lambdas(score_a, score_b, params, a_is_home, neutral)
    ga, gb = round(la), round(lb)
    if ga == gb and not math.isclose(la, lb):
        if la > lb:
            ga += 1
        else:
            gb += 1
    return int(ga), int(gb)


def sample_extra_time(
    score_a: float,
    score_b: float,
    params: ScoreModelParams,
    rng: np.random.Generator,
    neutral: bool = True,
) -> tuple[int, int]:
    """Sample extra-time goals over a 30-minute window."""
    la, lb = compute_lambdas(score_a, score_b, params, a_is_home=False, neutral=neutral)
    la *= params.et_rate_factor
    lb *= params.et_rate_factor
    return int(rng.poisson(la)), int(rng.poisson(lb))


def _pk_probability(score_a: float, score_b: float, params: ScoreModelParams) -> float:
    """Probability that side A wins a penalty shootout.

    Centred at 0.5; a small skill nudge proportional to the squared-score
    gap caps at ``pk_skill`` either way. Reflects the empirical reality
    that shootouts in elite men's football are very weakly predictable.
    """
    gap = (score_a - score_b) / 100.0
    return 0.5 + params.pk_skill * math.tanh(gap * 2.0)


def sample_penalties(
    score_a: float,
    score_b: float,
    params: ScoreModelParams,
    rng: np.random.Generator,
) -> WinnerSide:
    p = _pk_probability(score_a, score_b, params)
    return "A" if rng.random() < p else "B"


def deterministic_penalties(
    score_a: float, score_b: float, params: ScoreModelParams
) -> WinnerSide:
    """The slightly-favoured side wins; pure ties break toward A."""
    return "A" if _pk_probability(score_a, score_b, params) >= 0.5 else "B"


def simulate_match(
    score_a: float,
    score_b: float,
    params: ScoreModelParams,
    rng: np.random.Generator,
    knockout: bool,
    a_is_home: bool = False,
    neutral: bool = True,
) -> MatchOutcome:
    """Run a single sampled match. Knockouts cannot end as draws."""
    ga, gb = sample_regulation(score_a, score_b, params, rng, a_is_home, neutral)
    out = MatchOutcome(goals_a=ga, goals_b=gb)
    if knockout and ga == gb:
        out.went_to_et = True
        eta, etb = sample_extra_time(score_a, score_b, params, rng, neutral)
        out.et_goals_a = eta
        out.et_goals_b = etb
        if ga + eta == gb + etb:
            out.went_to_pk = True
            out.pk_winner = sample_penalties(score_a, score_b, params, rng)
    return out


def deterministic_match(
    score_a: float,
    score_b: float,
    params: ScoreModelParams,
    knockout: bool,
    a_is_home: bool = False,
    neutral: bool = True,
) -> MatchOutcome:
    """Single deterministic match outcome.

    For knockouts that come out tied on the deterministic score, the
    expected-goals tiebreaker has already nudged a winner; if for some
    reason they remain level (rare integer collision), penalties go to
    the favourite by ``_pk_probability``.
    """
    ga, gb = deterministic_regulation(score_a, score_b, params, a_is_home, neutral)
    out = MatchOutcome(goals_a=ga, goals_b=gb)
    if knockout and ga == gb:
        # Determ-regulation already nudges, so this only fires on exact ties
        # in both score and lambda (e.g. mirror-image matchup).
        out.went_to_pk = True
        out.pk_winner = deterministic_penalties(score_a, score_b, params)
    return out
