"""Tournament simulation: deterministic walk + Monte Carlo aggregation."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .score_model import ScoreModelParams
from .tournament import (
    GROUP_LETTERS,
    SimulatedTournament,
    play_tournament,
)

LOGGER = logging.getLogger(__name__)

KNOCKOUT_ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Champion"]
RANK_ORDER = ["Group", *KNOCKOUT_ROUNDS]


@dataclass
class MonteCarloSummary:
    """Aggregated probabilities across MC iterations, plus the deterministic run."""

    iterations: int
    probabilities: pd.DataFrame  # one row per team, columns = round reach probs
    champion_distribution: pd.Series  # P(champion) per team
    medals: pd.DataFrame  # team -> (P(gold), P(silver), P(bronze))
    deterministic: SimulatedTournament
    notes: list[str] = field(default_factory=list)


def _load_groups(path: str | Path, team_aliases: dict[str, str]) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    if not {"group", "team"}.issubset(df.columns):
        raise ValueError(f"Groups CSV {path!s} needs 'group' and 'team' columns.")
    if team_aliases:
        df["team"] = df["team"].replace(team_aliases)
    groups: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        groups[str(row["group"]).strip()].append(str(row["team"]).strip())
    for letter, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(
                f"Group {letter} has {len(teams)} teams; expected 4."
            )
    if set(groups) != set(GROUP_LETTERS):
        missing = set(GROUP_LETTERS) - set(groups)
        extra = set(groups) - set(GROUP_LETTERS)
        raise ValueError(
            f"Groups CSV must define {sorted(GROUP_LETTERS)}; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return dict(groups)


def _scores_from_rankings(rankings_csv: str | Path) -> dict[str, float]:
    df = pd.read_csv(rankings_csv)
    if not {"team", "final_score"}.issubset(df.columns):
        raise ValueError(
            f"Rankings CSV {rankings_csv!s} must have 'team' and 'final_score' columns."
        )
    return dict(zip(df["team"], df["final_score"].astype(float)))


def _team_reached(t: SimulatedTournament) -> dict[str, str]:
    """For each team that played, return the furthest round they reached."""
    reached: dict[str, str] = {}
    for letter, standing in t.group_standings.items():
        for r in standing:
            reached[r.team] = "Group"

    advancer_names = {r.team for r in t.advancers}
    for name in advancer_names:
        reached[name] = "R32"

    for m in t.knockout_matches:
        if m.outcome is None:
            continue
        if m.round_name == "3rd-place":
            continue
        winner = m.winner_team
        loser = m.team_b if m.outcome.winner == "A" else m.team_a
        # Loser's furthest reached is the round they played in.
        # Winner advances; we'll record below.
        reached[loser] = m.round_name
        # Winner of Final == Champion; winners of earlier rounds advance.
        if m.round_name == "Final":
            reached[winner] = "Champion"
        else:
            reached[winner] = _next_round(m.round_name)
    return reached


def _next_round(name: str) -> str:
    mapping = {"R32": "R16", "R16": "QF", "QF": "SF", "SF": "Final", "Final": "Champion"}
    return mapping[name]


def run(
    rankings_csv: str | Path,
    groups_csv: str | Path,
    iterations: int = 10_000,
    seed: Optional[int] = None,
    params: Optional[ScoreModelParams] = None,
    team_aliases: Optional[dict[str, str]] = None,
) -> MonteCarloSummary:
    """Run the deterministic walk plus ``iterations`` Monte Carlo simulations."""
    params = params or ScoreModelParams()
    team_aliases = team_aliases or {}

    groups = _load_groups(groups_csv, team_aliases)
    scores = _scores_from_rankings(rankings_csv)

    # Sanity: every team in the groups should be in the scores.
    missing = [t for ts in groups.values() for t in ts if t not in scores]
    if missing:
        raise ValueError(
            f"Teams missing from rankings CSV: {sorted(set(missing))}. "
            "Run `worldcup-ranker rank` first or check team_aliases."
        )

    # Deterministic walk.
    determ_rng = np.random.default_rng(seed if seed is not None else 0)
    deterministic = play_tournament(groups, scores, params, determ_rng, deterministic=True)

    # MC aggregation.
    mc_rng = np.random.default_rng(seed if seed is not None else 0)
    round_counts: dict[str, Counter[str]] = {
        r: Counter() for r in RANK_ORDER
    }
    medals = Counter()
    silvers = Counter()
    bronzes = Counter()
    finalists = Counter()

    for _ in range(iterations):
        t = play_tournament(groups, scores, params, mc_rng, deterministic=False)
        reached = _team_reached(t)
        for team, furthest in reached.items():
            round_counts[furthest][team] += 1
        if t.champion:
            medals[t.champion] += 1
        if t.runner_up:
            silvers[t.runner_up] += 1
            finalists[t.runner_up] += 1
        if t.champion:
            finalists[t.champion] += 1
        if t.third_place:
            bronzes[t.third_place] += 1

    all_teams = sorted({t for ts in groups.values() for t in ts})

    # Convert per-round "furthest reached" counts into cumulative
    # "at least reached round X" probabilities for clarity.
    cum = {team: {r: 0 for r in RANK_ORDER} for team in all_teams}
    for r in RANK_ORDER:
        for team, n in round_counts[r].items():
            cum[team][r] = n
    # Cumulative: P(team reached at least R32) = sum of P(furthest>=R32).
    prob_rows = []
    for team in all_teams:
        row = {"team": team}
        running = 0
        # Walk from "Champion" back to "Group" to compute "reached at least X".
        # P(at least R32) = P(furthest is R32) + P(R16) + P(QF) + ...
        for r in reversed(RANK_ORDER):
            running += cum[team][r]
            row[f"P_at_least_{r}"] = running / iterations
        prob_rows.append(row)
    probabilities = pd.DataFrame(prob_rows).sort_values(
        "P_at_least_Champion", ascending=False
    ).reset_index(drop=True)

    champion_distribution = pd.Series(
        {t: medals[t] / iterations for t in all_teams},
        name="P_champion",
    ).sort_values(ascending=False)

    medals_df = pd.DataFrame(
        [
            {
                "team": t,
                "P_gold": medals[t] / iterations,
                "P_silver": silvers[t] / iterations,
                "P_bronze": bronzes[t] / iterations,
                "P_finalist": finalists[t] / iterations,
            }
            for t in all_teams
        ]
    ).sort_values(["P_gold", "P_silver", "P_bronze"], ascending=False).reset_index(drop=True)

    notes = [
        "R32 bracket assignment is an approximation of FIFA's Annex C (495-row lookup): "
        "advancers are seeded by (finishing position, points, GD, GF) and paired via the "
        "standard 32-team bracket order, with same-group rematches swapped out.",
        "Penalty shootouts are modeled as ~50/50 with a small skill nudge -- elite men's "
        "PKs are weakly predictable.",
        "Score parameters are hand-tuned (mean total 2.5, gap coefficient 2.0); fitting "
        "against the historical results CSV is a follow-up.",
    ]

    return MonteCarloSummary(
        iterations=iterations,
        probabilities=probabilities,
        champion_distribution=champion_distribution,
        medals=medals_df,
        deterministic=deterministic,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def render_deterministic_bracket(
    t: SimulatedTournament, output_dir: str | Path
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "deterministic_bracket.md"

    lines = ["# Deterministic bracket walk", ""]

    lines.append("## Group stage standings")
    for letter in GROUP_LETTERS:
        lines.append(f"\n### Group {letter}\n")
        lines.append("| pos | team | pld | w | d | l | gf | ga | gd | pts |")
        lines.append("|---:|------|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(t.group_standings[letter], start=1):
            lines.append(
                f"| {i} | {r.team} | {r.played} | {r.wins} | {r.draws} | {r.losses} | "
                f"{r.goals_for} | {r.goals_against} | {r.goal_difference} | {r.points} |"
            )

    lines.append("\n## Knockouts\n")
    by_round: dict[str, list] = defaultdict(list)
    for m in t.knockout_matches:
        by_round[m.round_name].append(m)
    for r in ["R32", "R16", "QF", "SF", "3rd-place", "Final"]:
        if r not in by_round:
            continue
        lines.append(f"\n### {r}\n")
        for m in by_round[r]:
            out = m.outcome
            score = f"{out.goals_a}-{out.goals_b}"
            extra = ""
            if out.went_to_et:
                extra = f" (ET: {out.et_goals_a}-{out.et_goals_b})"
            if out.went_to_pk:
                extra += f" -> PK winner: {m.team_a if out.pk_winner == 'A' else m.team_b}"
            lines.append(f"- {m.team_a} {score}{extra} {m.team_b}")

    lines.append("\n## Final standings\n")
    lines.append(f"- 🥇 **Champion**: {t.champion}")
    lines.append(f"- 🥈 Runner-up: {t.runner_up}")
    lines.append(f"- 🥉 Third place: {t.third_place}")
    lines.append(f"- Fourth place: {t.fourth_place}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_simulation_report(
    summary: MonteCarloSummary, output_dir: str | Path
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "simulation_report.md"

    lines = [
        "# Tournament simulation report",
        "",
        f"_{summary.iterations:,} Monte Carlo iterations + 1 deterministic walk_",
        "",
        "## Deterministic final four",
        "",
        f"- 🥇 **Champion**: {summary.deterministic.champion}",
        f"- 🥈 Runner-up: {summary.deterministic.runner_up}",
        f"- 🥉 Third: {summary.deterministic.third_place}",
        f"- 4th: {summary.deterministic.fourth_place}",
        "",
        "## Monte Carlo: top 10 by P(Champion)",
        "",
        "| rank | team | P(Champion) | P(Final) | P(SF) | P(QF) | P(R16) | P(R32) |",
        "|---:|------|------:|------:|------:|------:|------:|------:|",
    ]
    top10 = summary.probabilities.head(10)
    for i, row in enumerate(top10.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {row.team} | "
            f"{row.P_at_least_Champion*100:.1f}% | "
            f"{row.P_at_least_Final*100:.1f}% | "
            f"{row.P_at_least_SF*100:.1f}% | "
            f"{row.P_at_least_QF*100:.1f}% | "
            f"{row.P_at_least_R16*100:.1f}% | "
            f"{row.P_at_least_R32*100:.1f}% |"
        )

    lines += ["", "## Caveats", ""]
    for n in summary.notes:
        lines.append(f"- {n}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_outputs(summary: MonteCarloSummary, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["determ_bracket"] = render_deterministic_bracket(
        summary.deterministic, output_dir
    )
    paths["probabilities"] = output_dir / "monte_carlo_probabilities.csv"
    summary.probabilities.to_csv(paths["probabilities"], index=False)
    paths["medals"] = output_dir / "monte_carlo_medals.csv"
    summary.medals.to_csv(paths["medals"], index=False)
    paths["report"] = render_simulation_report(summary, output_dir)

    return paths
