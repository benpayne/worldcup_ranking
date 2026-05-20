"""End-to-end pipeline tests."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from worldcup_ranker.config import (
    AppConfig,
    DataConfig,
    EloConfig,
    GoalPerformanceConfig,
    OutputConfig,
    RecentFormConfig,
    SquadStrengthConfig,
    TournamentConfig,
    Weights,
)
from worldcup_ranker.ranking import run_ranking, write_outputs
from worldcup_ranker.report import render_report


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _make_config(
    tmp_path: Path,
    results_csv: Path,
    squad_csv: Path | None = None,
    qualified_csv: Path | None = None,
) -> AppConfig:
    return AppConfig(
        tournament=TournamentConfig(start_date="2026-06-11", form_window_days=730, min_matches=2),
        weights=Weights(elo=0.5, recent_form=0.25, goal_performance=0.15, squad_strength=0.10),
        elo=EloConfig(
            initial_rating=1500,
            home_advantage=65,
            k_base=30,
            importance={"FIFA World Cup": 4.0, "FIFA World Cup qualification": 2.0},
            default_importance=1.0,
            goal_diff_multiplier=True,
        ),
        recent_form=RecentFormConfig(half_life_days=365, opponent_adjusted=True),
        goal_performance=GoalPerformanceConfig(goal_diff_cap=5, attack_weight=0.5, defense_weight=0.5),
        squad_strength=SquadStrengthConfig(csv_path=str(squad_csv) if squad_csv else None),
        data=DataConfig(
            results_csv=str(results_csv),
            qualified_teams_csv=str(qualified_csv) if qualified_csv else None,
            team_aliases={},
        ),
        output=OutputConfig(top_n=3, directory=str(tmp_path / "outputs"), generate_chart=False),
    )


def test_run_ranking_end_to_end(tmp_path, tiny_matches):
    results = tmp_path / "results.csv"
    _write_csv(results, tiny_matches)

    config = _make_config(tmp_path, results)
    # Chile only plays once in the fixture (below the default drop threshold);
    # disable dropping here so the test exercises all four teams.
    config.tournament.drop_below_min_matches = False
    result = run_ranking(config)

    # 4 teams in the fixture.
    assert result.teams_ranked == 4
    assert set(result.rankings["team"]) == {"Brazil", "Argentina", "Uruguay", "Chile"}

    # Brazil should rank above Uruguay given their head-to-head dominance.
    rank = result.rankings.set_index("team")["rank"]
    assert rank["Brazil"] < rank["Uruguay"]

    # final_score must be in [0, 100].
    assert (result.rankings["final_score"] >= 0).all()
    assert (result.rankings["final_score"] <= 100).all()

    # Squad column should be NaN (no CSV) and a note should mention it.
    assert result.rankings["squad_strength_score"].isna().all()
    assert any("Squad-strength" in n for n in result.notes)

    # Effective weights should sum to 1 and exclude squad_strength.
    assert "squad_strength" not in result.weights_effective
    assert math.isclose(sum(result.weights_effective.values()), 1.0, abs_tol=1e-9)


def test_pipeline_strict_date_filter(tmp_path, tiny_matches):
    """Post-cutoff matches must not influence any score."""
    results_path = tmp_path / "results.csv"

    # Run once on the full fixture (which contains post-cutoff matches).
    _write_csv(results_path, tiny_matches)
    config = _make_config(tmp_path, results_path)
    r_with_future = run_ranking(config)

    # Now drop the post-cutoff matches from the input file and re-run.
    cutoff = pd.Timestamp(config.tournament.start_date)
    pre_only = tiny_matches.loc[tiny_matches["date"] < cutoff].copy()
    _write_csv(results_path, pre_only)
    r_pre_only = run_ranking(config)

    a = r_with_future.rankings.set_index("team")["final_score"].sort_index()
    b = r_pre_only.rankings.set_index("team")["final_score"].sort_index()
    pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_qualified_teams_filter(tmp_path, tiny_matches):
    results = tmp_path / "results.csv"
    qualified = tmp_path / "qualified.csv"
    _write_csv(results, tiny_matches)
    pd.DataFrame({"team": ["Brazil", "Argentina"]}).to_csv(qualified, index=False)

    config = _make_config(tmp_path, results, qualified_csv=qualified)
    result = run_ranking(config)

    assert result.teams_ranked == 2
    assert set(result.rankings["team"]) == {"Brazil", "Argentina"}


def test_per_team_renormalization_when_squad_missing_for_some_teams(tmp_path, tiny_matches):
    """A team missing one component should be scored on the renormalized
    remaining weights, not penalized as if the missing score were 0."""
    results = tmp_path / "results.csv"
    squad = tmp_path / "squad.csv"
    _write_csv(results, tiny_matches)
    # Squad data for only two of four teams.
    pd.DataFrame(
        {"team": ["Brazil", "Argentina"], "score": [900, 850]}
    ).to_csv(squad, index=False)

    config = _make_config(tmp_path, results, squad_csv=squad)
    config.tournament.drop_below_min_matches = False
    result = run_ranking(config)

    by_team = result.rankings.set_index("team")
    # Brazil and Argentina were supplied; Uruguay and Chile were not.
    assert not pd.isna(by_team.loc["Brazil", "squad_strength_score"])
    assert pd.isna(by_team.loc["Uruguay", "squad_strength_score"])
    assert pd.isna(by_team.loc["Chile", "squad_strength_score"])

    # Reconstruct each team's expected final_score by per-team renorm.
    weights = result.weights_effective
    for team in by_team.index:
        row = by_team.loc[team]
        weighted = 0.0
        active = 0.0
        for comp, col in [
            ("elo", "elo_score"),
            ("recent_form", "recent_form_score"),
            ("goal_performance", "goal_performance_score"),
            ("squad_strength", "squad_strength_score"),
        ]:
            if comp not in weights:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            weighted += weights[comp] * val
            active += weights[comp]
        expected = weighted / active if active > 0 else 0.0
        assert row["final_score"] == pytest.approx(expected, abs=1e-9), team

    # Notes should mention the renorm.
    assert any("Per-team weight renormalization" in n for n in result.notes)


def test_per_team_renormalization_advantages_missing_data_team_over_zero_impute(
    tmp_path, tiny_matches
):
    """Sanity-check direction: a team with missing squad data scores higher
    under per-team renorm than it would under 0-impute, all else equal."""
    results = tmp_path / "results.csv"
    squad = tmp_path / "squad.csv"
    _write_csv(results, tiny_matches)
    # Argentina has a strong squad; Brazil/Uruguay/Chile do not.
    pd.DataFrame({"team": ["Argentina"], "score": [1000]}).to_csv(squad, index=False)

    config = _make_config(tmp_path, results, squad_csv=squad)
    config.tournament.drop_below_min_matches = False
    result = run_ranking(config)

    by_team = result.rankings.set_index("team")
    weights = result.weights_effective

    # Reconstruct the alternative 0-impute final_score for a missing-squad team.
    brazil = by_team.loc["Brazil"]
    zero_impute = (
        weights["elo"] * brazil["elo_score"]
        + weights["recent_form"] * brazil["recent_form_score"]
        + weights["goal_performance"] * brazil["goal_performance_score"]
        + weights["squad_strength"] * 0.0
    )
    assert brazil["final_score"] > zero_impute


def test_squad_strength_included_when_csv_supplied(tmp_path, tiny_matches):
    results = tmp_path / "results.csv"
    squad = tmp_path / "squad.csv"
    _write_csv(results, tiny_matches)
    pd.DataFrame(
        {
            "team": ["Brazil", "Argentina", "Uruguay", "Chile"],
            "score": [900, 850, 600, 400],
        }
    ).to_csv(squad, index=False)

    config = _make_config(tmp_path, results, squad_csv=squad)
    result = run_ranking(config)

    assert "squad_strength" in result.weights_effective
    assert math.isclose(sum(result.weights_effective.values()), 1.0, abs_tol=1e-9)
    assert result.rankings["squad_strength_score"].notna().all()


def test_write_outputs_creates_files(tmp_path, tiny_matches):
    results = tmp_path / "results.csv"
    _write_csv(results, tiny_matches)
    config = _make_config(tmp_path, results)

    result = run_ranking(config)
    paths = write_outputs(result, config.output.directory, top_n=config.output.top_n)
    assert paths["full"].exists()
    assert paths["top"].exists()

    top_df = pd.read_csv(paths["top"])
    assert len(top_df) == config.output.top_n

    # Report renders without error.
    report_path = render_report(result, config, config.output.directory)
    assert report_path.exists()
    text = report_path.read_text()
    assert "Top 3" in text
    assert "Caveats" in text


def _low_sample_fixture() -> list[dict]:
    """Chile plays only once; Brazil/Argentina have three each."""
    return [
        {
            "date": pd.Timestamp("2025-06-05"),
            "home_team": "Brazil",
            "away_team": "Chile",
            "home_score": 4,
            "away_score": 0,
            "tournament": "Friendly",
            "city": "x",
            "country": "x",
            "neutral": False,
        },
        {
            "date": pd.Timestamp("2025-08-01"),
            "home_team": "Brazil",
            "away_team": "Argentina",
            "home_score": 1,
            "away_score": 1,
            "tournament": "Friendly",
            "city": "x",
            "country": "x",
            "neutral": True,
        },
        {
            "date": pd.Timestamp("2025-11-01"),
            "home_team": "Argentina",
            "away_team": "Brazil",
            "home_score": 2,
            "away_score": 2,
            "tournament": "Friendly",
            "city": "x",
            "country": "x",
            "neutral": True,
        },
        {
            "date": pd.Timestamp("2026-02-15"),
            "home_team": "Brazil",
            "away_team": "Argentina",
            "home_score": 3,
            "away_score": 0,
            "tournament": "Friendly",
            "city": "x",
            "country": "x",
            "neutral": True,
        },
    ]


def test_min_matches_drops_low_sample_teams_by_default(tmp_path):
    """Default behavior: Chile (1 match) is excluded from the output."""
    results = tmp_path / "results.csv"
    pd.DataFrame(_low_sample_fixture()).to_csv(results, index=False)

    config = _make_config(tmp_path, results)
    config.tournament.min_matches = 3  # drop_below_min_matches=True by default
    result = run_ranking(config)

    assert "Chile" not in set(result.rankings["team"])
    assert any("Excluded" in n and "Chile" in n for n in result.notes)


def test_min_matches_keeps_and_annotates_when_drop_disabled(tmp_path):
    """Opt-in legacy behavior: keep low-sample teams and tag them."""
    results = tmp_path / "results.csv"
    pd.DataFrame(_low_sample_fixture()).to_csv(results, index=False)

    config = _make_config(tmp_path, results)
    config.tournament.min_matches = 3
    config.tournament.drop_below_min_matches = False
    result = run_ranking(config)

    assert any("fewer than 3 matches" in n for n in result.notes)
    chile_row = result.rankings.set_index("team").loc["Chile"]
    assert chile_row["notes"] == "<3 matches in window"
