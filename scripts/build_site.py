"""Assemble a static site from the most recent ``outputs/`` directory.

Reads ``outputs/model_report.md`` and renders it as an ``index.html`` page
with the top-N chart embedded at the top and download links for the CSVs.
Run after ``worldcup-ranker rank``.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import markdown
except ImportError as exc:  # pragma: no cover - friendly hint
    raise SystemExit(
        "The 'markdown' package is required. Install with: pip install -e '.[site]'"
    ) from exc


OUTPUTS = Path("outputs")
SITE = Path("site")
ASSETS = (
    "top_24_rankings.csv",
    "full_rankings.csv",
    "model_report.md",
    "top_24.png",
    # Tournament simulator outputs (present when `worldcup-ranker simulate` ran).
    "simulation_report.md",
    "deterministic_bracket.md",
    "monte_carlo_probabilities.csv",
    "monte_carlo_medals.csv",
)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>World Cup Strength Ranking</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1rem 4rem;
    color: #1f2328;
    background: #fff;
    line-height: 1.55;
  }}
  h1, h2, h3 {{ line-height: 1.25; }}
  h1 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #eaecef; padding-bottom: 0.2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 0.45rem 0.7rem; border-bottom: 1px solid #eaecef; }}
  th {{ background: #f6f8fa; text-align: left; }}
  /* numeric columns right-aligned */
  td:nth-child(n+3), th:nth-child(n+3) {{ text-align: right; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto; }}
  code {{ background: #f6f8fa; padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.95em; }}
  .meta {{ color: #57606a; font-size: 0.9rem; }}
  .downloads {{
    margin: 1rem 0;
    padding: 0.75rem 1rem;
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 8px;
  }}
  .downloads a {{ margin-right: 1rem; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0d1117; color: #c9d1d9; }}
    h1 {{ border-color: #30363d; }}
    h2 {{ border-color: #21262d; }}
    th {{ background: #161b22; }}
    th, td {{ border-color: #21262d; }}
    code, .downloads {{ background: #161b22; border-color: #30363d; }}
    .meta {{ color: #8b949e; }}
    a {{ color: #58a6ff; }}
  }}
</style>
</head>
<body>
<p class="meta">Built {built_at}.</p>
<div class="downloads">
  <strong>Downloads:</strong>
  <a href="top_24_rankings.csv">top_24_rankings.csv</a>
  <a href="full_rankings.csv">full_rankings.csv</a>
  <a href="model_report.md">model_report.md</a>
  {simulation_links}
</div>
{chart}
{body}
{simulation_section}
</body>
</html>
"""


def main() -> int:
    if not OUTPUTS.exists():
        sys.stderr.write(
            "outputs/ does not exist. Run `worldcup-ranker rank` first.\n"
        )
        return 1

    SITE.mkdir(exist_ok=True)

    for name in ASSETS:
        src = OUTPUTS / name
        if src.exists():
            shutil.copy2(src, SITE / name)

    report_path = OUTPUTS / "model_report.md"
    if not report_path.exists():
        sys.stderr.write(f"Missing {report_path}; cannot build site.\n")
        return 1

    body_md = report_path.read_text(encoding="utf-8")
    body_html = markdown.markdown(body_md, extensions=["tables", "fenced_code"])

    chart_html = (
        '<img src="top_24.png" alt="Top 24 strength ranking">'
        if (SITE / "top_24.png").exists()
        else ""
    )
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sim_links = ""
    sim_section = ""
    sim_report = SITE / "simulation_report.md"
    if sim_report.exists():
        sim_links = (
            '<a href="simulation_report.md">simulation_report.md</a> '
            '<a href="deterministic_bracket.md">deterministic_bracket.md</a> '
            '<a href="monte_carlo_probabilities.csv">monte_carlo_probabilities.csv</a> '
            '<a href="monte_carlo_medals.csv">monte_carlo_medals.csv</a>'
        )
        sim_md = sim_report.read_text(encoding="utf-8")
        sim_html = markdown.markdown(
            sim_md, extensions=["tables", "fenced_code"]
        )
        sim_section = f'<hr>\n<section id="simulation">\n{sim_html}\n</section>'

    (SITE / "index.html").write_text(
        PAGE.format(
            body=body_html,
            chart=chart_html,
            built_at=built_at,
            simulation_links=sim_links,
            simulation_section=sim_section,
        ),
        encoding="utf-8",
    )
    print(f"Built {SITE / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
