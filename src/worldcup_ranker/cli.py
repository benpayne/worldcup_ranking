"""Command-line interface for worldcup-ranker."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .data_sources import DEFAULT_RESULTS_URL, fetch_results_csv
from .plotting import plot_top_n
from .ranking import run_ranking, write_outputs
from .report import render_report

LOGGER = logging.getLogger("worldcup_ranker")

DEFAULT_CONFIG = "config/default.yaml"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="worldcup-ranker")
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    sub = p.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch-data", help="Download the public match-results CSV."
    )
    fetch.add_argument(
        "--url", default=DEFAULT_RESULTS_URL, help="Override the source URL."
    )
    fetch.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing file."
    )

    rank = sub.add_parser("rank", help="Run the ranking pipeline.")
    rank.add_argument(
        "--start-date",
        help="Override tournament start date (YYYY-MM-DD).",
    )
    rank.add_argument(
        "--qualified-teams",
        help="Optional CSV listing qualified teams (column 'team').",
    )
    rank.add_argument(
        "--results-csv",
        help="Override path to international match results CSV.",
    )
    rank.add_argument(
        "--squad-csv",
        help="Optional path to a squad-strength CSV (columns 'team','score').",
    )
    rank.add_argument(
        "--xg-csv",
        help="Optional path to a precomputed xG CSV (build with `statsbomb build`).",
    )
    rank.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip generating the top-N PNG chart.",
    )

    sub.add_parser(
        "report",
        help="Re-render the model report from the most recent rankings.",
    )

    sb = sub.add_parser(
        "statsbomb",
        help="Build xG / squad-strength enrichment CSVs from a local clone "
        "of statsbomb/open-data.",
    )
    sb_sub = sb.add_subparsers(dest="sb_command", required=True)
    sb_build = sb_sub.add_parser(
        "build",
        help="Build both the xG CSV and the squad-strength CSV.",
    )
    sb_build.add_argument(
        "--open-data-path",
        required=True,
        help="Path to a local clone of github.com/statsbomb/open-data.",
    )
    sb_build.add_argument(
        "--xg-out",
        default="data/processed/statsbomb_xg.csv",
        help="Where to write the per-match xG CSV.",
    )
    sb_build.add_argument(
        "--squad-out",
        default="data/processed/statsbomb_squad.csv",
        help="Where to write the per-team squad-strength CSV.",
    )
    sb_build.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Lookback for the squad CSV (defaults to tournament.form_window_days).",
    )
    sb_build.add_argument(
        "--top-n-players",
        type=int,
        default=23,
        help="Number of players summed per team for squad strength.",
    )
    sb_build.add_argument(
        "--start-date",
        help="Cutoff date for the squad CSV (defaults to tournament.start_date).",
    )
    sb_build.add_argument(
        "--skip-xg",
        action="store_true",
        help="Skip building the xG CSV.",
    )
    sb_build.add_argument(
        "--skip-squad",
        action="store_true",
        help="Skip building the squad CSV.",
    )

    return p


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _apply_overrides(config, args):
    if getattr(args, "start_date", None):
        config.tournament.start_date = args.start_date
    if getattr(args, "qualified_teams", None):
        config.data.qualified_teams_csv = args.qualified_teams
    if getattr(args, "results_csv", None):
        config.data.results_csv = args.results_csv
    if getattr(args, "squad_csv", None):
        config.squad_strength.csv_path = args.squad_csv
    if getattr(args, "xg_csv", None):
        config.goal_performance.xg_csv = args.xg_csv
    if getattr(args, "no_chart", False):
        config.output.generate_chart = False
    return config


def cmd_fetch(args, config) -> int:
    target = Path(config.data.results_csv)
    fetch_results_csv(target, url=args.url, overwrite=args.overwrite)
    print(f"Saved match results to {target}")
    return 0


def cmd_rank(args, config) -> int:
    config = _apply_overrides(config, args)
    result = run_ranking(config)
    paths = write_outputs(result, config.output.directory, config.output.top_n)
    report_path = render_report(result, config, config.output.directory)
    print(f"Top-{config.output.top_n} rankings: {paths['top']}")
    print(f"Full rankings:        {paths['full']}")
    print(f"Model report:         {report_path}")
    if config.output.generate_chart:
        try:
            chart_path = plot_top_n(result, config.output.directory, config.output.top_n)
            print(f"Chart:                {chart_path}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Chart generation failed: %s", exc)
    return 0


def cmd_report(args, config) -> int:
    # Re-run the pipeline (cheap) and re-render the report.
    result = run_ranking(config)
    path = render_report(result, config, config.output.directory)
    print(f"Wrote {path}")
    return 0


def cmd_statsbomb(args, config) -> int:
    if args.sb_command != "build":
        raise SystemExit(f"Unknown statsbomb sub-command: {args.sb_command}")

    from .statsbomb import build_squad_csv, build_xg_csv

    import pandas as pd

    aliases = config.data.team_aliases or {}
    open_data_path = args.open_data_path

    if not args.skip_xg:
        xg_path = build_xg_csv(open_data_path, args.xg_out, team_aliases=aliases)
        print(f"xG CSV:    {xg_path}")
    else:
        print("xG CSV:    skipped (--skip-xg)")

    if not args.skip_squad:
        cutoff = pd.Timestamp(
            args.start_date if args.start_date else config.tournament.start_date
        )
        lookback = (
            args.lookback_days
            if args.lookback_days is not None
            else config.tournament.form_window_days
        )
        squad_path = build_squad_csv(
            open_data_path,
            args.squad_out,
            cutoff=cutoff,
            lookback_days=lookback,
            top_n_players=args.top_n_players,
            team_aliases=aliases,
        )
        print(f"Squad CSV: {squad_path}")
    else:
        print("Squad CSV: skipped (--skip-squad)")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    config = load_config(args.config)

    if args.command == "fetch-data":
        return cmd_fetch(args, config)
    if args.command == "rank":
        return cmd_rank(args, config)
    if args.command == "report":
        return cmd_report(args, config)
    if args.command == "statsbomb":
        return cmd_statsbomb(args, config)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
