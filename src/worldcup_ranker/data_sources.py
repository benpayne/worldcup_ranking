"""Data loading helpers.

All data sources used here are public/free. The primary source is the
"International football results from 1872 to 2026" dataset by martj42,
distributed on Kaggle and GitHub under CC0:
    https://github.com/martj42/international_results
    https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

If automated Kaggle authentication is not configured the user can download
``results.csv`` manually and place it at ``data/raw/results.csv``.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Public mirror of martj42's results.csv (the upstream file is updated weekly).
DEFAULT_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

EXPECTED_RESULT_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}


@dataclass(frozen=True)
class DataSourceInfo:
    """Lightweight record describing a data source used in this run."""

    name: str
    url: str
    licence: str
    notes: str = ""


PRIMARY_SOURCE = DataSourceInfo(
    name="martj42/international_results",
    url="https://github.com/martj42/international_results",
    licence="CC0 1.0 (public domain)",
    notes="Match-level results since 1872; updated frequently.",
)


def fetch_results_csv(
    target_path: str | Path,
    url: str = DEFAULT_RESULTS_URL,
    overwrite: bool = False,
) -> Path:
    """Download the public match-results CSV to ``target_path``.

    Network access is required. If the user does not have outbound access they
    should download the file manually and drop it into ``data/raw/results.csv``.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        LOGGER.info("Results CSV already exists at %s; skipping download.", target)
        return target

    LOGGER.info("Downloading match results from %s ...", url)
    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        with target.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    LOGGER.info("Saved match results to %s", target)
    return target


def load_results(
    path: str | Path,
    team_aliases: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load and normalize the results CSV.

    Returns a DataFrame with parsed dates and canonicalized team names. The
    DataFrame is sorted by date ascending. Rows with missing scores or dates
    are dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Match results CSV not found at {path!s}. Run "
            "`worldcup-ranker fetch-data` or download results.csv manually."
        )

    df = pd.read_csv(path)
    missing = EXPECTED_RESULT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Results CSV is missing expected columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    aliases = team_aliases or {}
    if aliases:
        df["home_team"] = df["home_team"].replace(aliases)
        df["away_team"] = df["away_team"].replace(aliases)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_qualified_teams(
    path: Optional[str | Path],
    team_aliases: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Load an optional list of teams to score.

    The CSV must have a single column named ``team`` (or be a one-column CSV).
    Returns ``None`` if no path is supplied.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Qualified-teams CSV not found at {p!s}")

    df = pd.read_csv(p)
    if "team" in df.columns:
        teams = df["team"].dropna().astype(str).tolist()
    else:
        teams = df.iloc[:, 0].dropna().astype(str).tolist()

    aliases = team_aliases or {}
    teams = [aliases.get(t, t) for t in teams]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for t in teams:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def load_squad_strength(
    path: Optional[str | Path],
    team_aliases: Optional[dict[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Load an optional squad-strength CSV.

    Expected columns: ``team``, ``score`` (any numeric scale; will be
    rescaled to 0-100 downstream). Returns ``None`` if ``path`` is None or
    the file does not exist (the squad-strength component is then skipped
    and remaining weights are renormalized).
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        LOGGER.warning(
            "Squad-strength CSV %s not found; skipping squad component.", p
        )
        return None

    df = pd.read_csv(p)
    if not {"team", "score"}.issubset(df.columns):
        raise ValueError(
            f"Squad-strength CSV {p!s} must have columns 'team' and 'score'."
        )
    aliases = team_aliases or {}
    if aliases:
        df["team"] = df["team"].replace(aliases)
    df = df.dropna(subset=["team", "score"]).drop_duplicates("team", keep="last")
    return df.reset_index(drop=True)


def filter_before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return only rows strictly before the tournament start date.

    Centralized to make it easy to audit that no future data leaks into the
    ranking.
    """
    return df.loc[df["date"] < cutoff].copy()


def restrict_to_teams(
    df: pd.DataFrame, teams: Iterable[str]
) -> pd.DataFrame:
    """Return matches in which at least one of ``teams`` participated."""
    team_set = set(teams)
    mask = df["home_team"].isin(team_set) | df["away_team"].isin(team_set)
    return df.loc[mask].copy()
