"""Shared fixtures for the test suite."""
from __future__ import annotations

import pandas as pd
import pytest


def _row(date, home, away, hs, as_, tournament="Friendly", neutral=False):
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": tournament,
        "city": "Anywhere",
        "country": "Anywhere",
        "neutral": neutral,
    }


@pytest.fixture
def tiny_matches() -> pd.DataFrame:
    """A small set of matches spanning the 2026 cutoff.

    Used to verify both Elo math and date-filter correctness.
    """
    rows = [
        # before cutoff
        _row("2025-01-15", "Brazil", "Argentina", 2, 1, "FIFA World Cup qualification"),
        _row("2025-03-10", "Argentina", "Uruguay", 1, 1, "FIFA World Cup qualification"),
        _row("2025-06-05", "Brazil", "Chile", 4, 0, "Friendly"),
        _row("2025-09-12", "Uruguay", "Brazil", 0, 2, "FIFA World Cup qualification"),
        _row("2026-01-15", "Argentina", "Brazil", 1, 0, "Friendly"),
        _row("2026-03-22", "Brazil", "Uruguay", 3, 1, "Friendly"),
        # after cutoff -- must never influence Elo / form
        _row("2026-06-15", "Brazil", "Argentina", 0, 5, "FIFA World Cup"),
        _row("2026-07-01", "Argentina", "Uruguay", 5, 0, "FIFA World Cup"),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    return df
