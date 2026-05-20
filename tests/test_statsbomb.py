"""Tests for the StatsBomb open-data adapter."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from worldcup_ranker.statsbomb import (
    build_squad_csv,
    build_xg_csv,
    iter_mens_international_matches,
    load_xg_csv,
)


FIXTURE = Path(__file__).parent / "fixtures" / "statsbomb"


def test_iter_filters_to_mens_internationals():
    matches = list(iter_mens_international_matches(FIXTURE))
    # Euro 2024 has 2 matches in our fixture; Bundesliga is excluded.
    assert {m.match_id for m in matches} == {1001, 1002}
    assert all(m.competition == "UEFA Euro" for m in matches)


def test_iter_missing_competitions_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_mens_international_matches(tmp_path))


def test_build_xg_csv_aggregates_per_team_per_match(tmp_path):
    out = tmp_path / "xg.csv"
    build_xg_csv(FIXTURE, out)
    df = pd.read_csv(out)

    assert set(df.columns) >= {
        "date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "competition",
        "season",
        "match_id",
    }

    by_id = df.set_index("match_id")
    # Match 1001: Spain (0.45 + 0.12) vs Croatia (0.08 + 0.22)
    assert by_id.loc[1001, "home_xg"] == pytest.approx(0.57)
    assert by_id.loc[1001, "away_xg"] == pytest.approx(0.30)
    # Match 1002: Spain 0.30 vs Italy (0.55 + 0.18)
    assert by_id.loc[1002, "home_xg"] == pytest.approx(0.30)
    assert by_id.loc[1002, "away_xg"] == pytest.approx(0.73)


def test_build_xg_csv_applies_aliases(tmp_path):
    out = tmp_path / "xg.csv"
    build_xg_csv(FIXTURE, out, team_aliases={"Spain": "ESP"})
    df = pd.read_csv(out)
    assert "ESP" in set(df["home_team"]).union(df["away_team"])
    assert "Spain" not in set(df["home_team"]).union(df["away_team"])


def test_build_squad_csv_credits_shooter_and_key_pass(tmp_path):
    out = tmp_path / "squad.csv"
    # Cutoff just after the fixture matches.
    cutoff = pd.Timestamp("2024-07-01")
    build_squad_csv(
        FIXTURE,
        out,
        cutoff=cutoff,
        lookback_days=365,
        top_n_players=23,
    )
    df = pd.read_csv(out)

    # Morata: 0.45 (shot, match 1001) + 0.30 (shot, match 1002) = 0.75
    # Yamal:  0.12
    # Ruiz:   0.45 (key-pass assist on Morata's shot)
    # Spain top-3 = 0.75 + 0.45 + 0.12 = 1.32
    spain = df.set_index("team").loc["Spain", "score"]
    assert spain == pytest.approx(1.32, abs=1e-4)


def test_build_squad_csv_excludes_matches_outside_window(tmp_path):
    out = tmp_path / "squad.csv"
    # Cutoff before any fixture match -> zero coverage.
    cutoff = pd.Timestamp("2024-01-01")
    build_squad_csv(FIXTURE, out, cutoff=cutoff, lookback_days=365)
    df = pd.read_csv(out)
    assert df.empty


def test_load_xg_csv_normalises_dates_and_aliases(tmp_path):
    out = tmp_path / "xg.csv"
    build_xg_csv(FIXTURE, out)
    loaded = load_xg_csv(out, team_aliases={"Spain": "ESP"})
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    assert "ESP" in set(loaded["home_team"]).union(loaded["away_team"])
