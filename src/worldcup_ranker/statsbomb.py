"""Build StatsBomb-derived enrichment CSVs from a local clone of
``statsbomb/open-data``.

Coverage note
-------------
StatsBomb open data covers a small set of men's national-team
tournaments (recent World Cups, Euros, Copa América). For matches
outside that set the downstream features fall back to the goals-based
proxy already implemented in :mod:`worldcup_ranker.features`.

Why not statsbombpy?
--------------------
``statsbombpy`` is built around remote / commercial-API access and has
no first-class mode for reading from a local clone. The open-data JSON
schema is small enough to parse in-place, so we read the files directly
and avoid the extra dependency.

Licence
-------
StatsBomb Open Data User Agreement — free for non-commercial use with
attribution. See ``https://github.com/statsbomb/open-data``.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

OPEN_DATA_URL = "https://github.com/statsbomb/open-data"
OPEN_DATA_LICENCE = (
    "StatsBomb Open Data User Agreement (non-commercial use with attribution)"
)


@dataclass(frozen=True)
class _Match:
    match_id: int
    date: pd.Timestamp
    home_team: str
    away_team: str
    competition: str
    season: str


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_mens_international(competition: dict) -> bool:
    """True if this competition is a men's national-team tournament."""
    name = (competition.get("competition_name") or "").lower()
    international = bool(competition.get("competition_international", False))
    gender = (competition.get("competition_gender") or "male").lower()
    if gender != "male" or not international:
        return False
    return any(
        keyword in name
        for keyword in ("world cup", "euro", "copa america", "copa américa")
    )


def iter_mens_international_matches(open_data_path: Path) -> Iterator[_Match]:
    """Yield every men's national-team match present in the clone."""
    comps_path = open_data_path / "data" / "competitions.json"
    if not comps_path.exists():
        raise FileNotFoundError(
            f"Missing {comps_path}. Clone the dataset first: "
            f"git clone --depth 1 {OPEN_DATA_URL}"
        )
    competitions = _load_json(comps_path)
    if not isinstance(competitions, list):
        raise ValueError(f"{comps_path} did not contain a JSON array.")

    for comp in competitions:
        if not _is_mens_international(comp):
            continue
        cid = comp["competition_id"]
        sid = comp["season_id"]
        matches_path = open_data_path / "data" / "matches" / str(cid) / f"{sid}.json"
        if not matches_path.exists():
            LOGGER.debug("No matches file at %s; skipping.", matches_path)
            continue
        matches = _load_json(matches_path)
        for m in matches:
            yield _Match(
                match_id=int(m["match_id"]),
                date=pd.to_datetime(m["match_date"]),
                home_team=m["home_team"]["home_team_name"],
                away_team=m["away_team"]["away_team_name"],
                competition=comp["competition_name"],
                season=comp["season_name"],
            )


def build_xg_csv(
    open_data_path: str | Path,
    output_csv: str | Path,
    team_aliases: Optional[dict[str, str]] = None,
) -> Path:
    """Aggregate per-match xG and write a CSV ready for ``--xg-csv``.

    Output columns:
        date, home_team, away_team, home_xg, away_xg,
        competition, season, match_id
    """
    open_data_path = Path(open_data_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aliases = team_aliases or {}

    rows = []
    n_matches = 0
    n_skipped = 0
    for match in iter_mens_international_matches(open_data_path):
        events_path = open_data_path / "data" / "events" / f"{match.match_id}.json"
        if not events_path.exists():
            n_skipped += 1
            continue
        events = _load_json(events_path)
        home_xg = 0.0
        away_xg = 0.0
        for ev in events:
            shot = ev.get("shot")
            if not shot:
                continue
            xg = shot.get("statsbomb_xg")
            if xg is None:
                continue
            team = (ev.get("team") or {}).get("name", "")
            if team == match.home_team:
                home_xg += float(xg)
            elif team == match.away_team:
                away_xg += float(xg)
        rows.append(
            {
                "date": match.date,
                "home_team": aliases.get(match.home_team, match.home_team),
                "away_team": aliases.get(match.away_team, match.away_team),
                "home_xg": round(home_xg, 4),
                "away_xg": round(away_xg, 4),
                "competition": match.competition,
                "season": match.season,
                "match_id": match.match_id,
            }
        )
        n_matches += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    LOGGER.info(
        "Wrote %d match xG rows to %s (skipped %d matches missing events file)",
        n_matches,
        output_csv,
        n_skipped,
    )
    return output_csv


def build_squad_csv(
    open_data_path: str | Path,
    output_csv: str | Path,
    cutoff: pd.Timestamp,
    lookback_days: int = 730,
    top_n_players: int = 23,
    team_aliases: Optional[dict[str, str]] = None,
) -> Path:
    """Build a per-team squad-strength CSV from player xG + xA contributions.

    For each player who appears in any covered match within the look-back
    window, sum the xG of shots they took (finishing) plus the xG of
    shots they key-passed (creating). For each national team, sum the
    top ``top_n_players`` such totals. The resulting CSV has columns
    ``team`` and ``score`` and plugs straight into ``--squad-csv``.
    """
    open_data_path = Path(open_data_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aliases = team_aliases or {}
    window_start = cutoff - pd.Timedelta(days=lookback_days)

    # team -> player -> xG+xA total
    contributions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    n_matches = 0
    for match in iter_mens_international_matches(open_data_path):
        if not (window_start <= match.date < cutoff):
            continue
        events_path = open_data_path / "data" / "events" / f"{match.match_id}.json"
        if not events_path.exists():
            continue
        events = _load_json(events_path)

        # Index events by id so we can resolve key-pass links cheaply.
        events_by_id: dict[str, dict] = {ev["id"]: ev for ev in events if "id" in ev}

        for ev in events:
            shot = ev.get("shot")
            if not shot:
                continue
            xg = shot.get("statsbomb_xg")
            if xg is None:
                continue
            xg = float(xg)

            # Credit the shooter (xG / finishing).
            team = (ev.get("team") or {}).get("name", "")
            player = (ev.get("player") or {}).get("name", "")
            if team and player:
                canon_team = aliases.get(team, team)
                contributions[canon_team][player] += xg

            # Credit the key-pass passer (xA proxy / creating).
            kp_id = shot.get("key_pass_id")
            if kp_id and kp_id in events_by_id:
                pe = events_by_id[kp_id]
                p_team = (pe.get("team") or {}).get("name", "")
                p_player = (pe.get("player") or {}).get("name", "")
                if p_team and p_player:
                    canon_p_team = aliases.get(p_team, p_team)
                    contributions[canon_p_team][p_player] += xg

        n_matches += 1

    rows = []
    for team, totals in contributions.items():
        top = sorted(totals.values(), reverse=True)[:top_n_players]
        rows.append({"team": team, "score": round(sum(top), 4)})

    df = pd.DataFrame(rows, columns=["team", "score"])
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    LOGGER.info(
        "Wrote %d team rows to %s from %d covered matches "
        "(window: %d days ending %s)",
        len(rows),
        output_csv,
        n_matches,
        lookback_days,
        cutoff.date(),
    )
    return output_csv


def load_xg_csv(
    path: str | Path,
    team_aliases: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load an xG CSV produced by :func:`build_xg_csv`.

    Returns a DataFrame keyed on (date, home_team, away_team) with
    ``home_xg`` and ``away_xg``. Suitable for merging into a per-match
    view.
    """
    df = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_xg", "away_xg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"xG CSV {path!s} missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    aliases = team_aliases or {}
    if aliases:
        df["home_team"] = df["home_team"].replace(aliases)
        df["away_team"] = df["away_team"].replace(aliases)
    return df[["date", "home_team", "away_team", "home_xg", "away_xg"]]
