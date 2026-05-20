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
        "--no-chart",
        action="store_true",
        help="Skip generating the top-N PNG chart.",
    )

    sub.add_parser(
        "report",
        help="Re-render the model report from the most recent rankings.",
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
