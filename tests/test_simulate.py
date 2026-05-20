"""End-to-end tests for the simulator orchestration layer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from worldcup_ranker.score_model import ScoreModelParams
from worldcup_ranker.simulate import run, write_outputs
from worldcup_ranker.tournament import GROUP_LETTERS


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    teams = [f"{letter}{i}" for letter in GROUP_LETTERS for i in range(1, 5)]
    rankings = pd.DataFrame(
        {
            "team": teams,
            # Score order: highest seed in each group is strongest.
            "final_score": [
                85.0 - (i - 1) * 20
                for _ in GROUP_LETTERS
                for i in range(1, 5)
            ],
        }
    )
    rankings_csv = tmp_path / "rankings.csv"
    rankings.to_csv(rankings_csv, index=False)

    groups_rows = []
    for letter in GROUP_LETTERS:
        for i in range(1, 5):
            groups_rows.append({"group": letter, "team": f"{letter}{i}"})
    groups_csv = tmp_path / "groups.csv"
    pd.DataFrame(groups_rows).to_csv(groups_csv, index=False)

    return rankings_csv, groups_csv, teams


def test_run_produces_deterministic_plus_mc(tmp_path):
    rankings_csv, groups_csv, teams = _build_fixture(tmp_path)
    summary = run(
        rankings_csv=rankings_csv,
        groups_csv=groups_csv,
        iterations=200,
        seed=42,
    )

    # Deterministic walk is populated.
    assert summary.deterministic.champion is not None
    assert summary.deterministic.runner_up is not None

    # MC probabilities table covers every team in the field.
    assert set(summary.probabilities["team"]) == set(teams)

    # Every team's P(R32) should be in [0, 1] and P(Champion) <= P(R32).
    for _, row in summary.probabilities.iterrows():
        assert 0.0 <= row["P_at_least_R32"] <= 1.0
        assert row["P_at_least_Champion"] <= row["P_at_least_R32"]

    # MC champion probabilities sum to 1 over 200 iterations.
    assert summary.champion_distribution.sum() == pytest.approx(1.0, abs=1e-9)


def test_run_reproducible_with_seed(tmp_path):
    rankings_csv, groups_csv, _ = _build_fixture(tmp_path)
    a = run(rankings_csv, groups_csv, iterations=100, seed=7)
    b = run(rankings_csv, groups_csv, iterations=100, seed=7)
    pd.testing.assert_frame_equal(a.probabilities, b.probabilities)
    assert a.deterministic.champion == b.deterministic.champion


def test_run_missing_team_raises(tmp_path):
    rankings_csv, groups_csv, _ = _build_fixture(tmp_path)
    # Drop a team from the rankings CSV to trigger the safety check.
    df = pd.read_csv(rankings_csv)
    df = df[df["team"] != "A1"]
    df.to_csv(rankings_csv, index=False)
    with pytest.raises(ValueError, match="Teams missing"):
        run(rankings_csv, groups_csv, iterations=10, seed=0)


def test_write_outputs_creates_files(tmp_path):
    rankings_csv, groups_csv, _ = _build_fixture(tmp_path)
    summary = run(rankings_csv, groups_csv, iterations=50, seed=0)
    paths = write_outputs(summary, tmp_path / "sim-out")
    for k in ("determ_bracket", "probabilities", "medals", "report"):
        assert paths[k].exists(), k
    report_text = paths["report"].read_text()
    assert "Monte Carlo" in report_text
    assert "Champion" in report_text


def test_p_champion_concentrates_on_stronger_groups(tmp_path):
    """In our fixture, group winners are stronger than everyone else, so MC
    champions should overwhelmingly be group-winner teams (names ending in '1')."""
    rankings_csv, groups_csv, _ = _build_fixture(tmp_path)
    summary = run(rankings_csv, groups_csv, iterations=500, seed=99)
    by_team = summary.medals.set_index("team")
    total_gold_among_winners = by_team.loc[
        [f"{letter}1" for letter in GROUP_LETTERS], "P_gold"
    ].sum()
    # The fixture has group winners 20 points above runners-up -- meaningful
    # advantage but Poisson sampling still produces upsets, so set the bar at
    # "majority of champions come from the strongest team in their group".
    assert total_gold_among_winners > 0.6
