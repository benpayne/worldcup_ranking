"""Assemble a static site from the most recent ``outputs/`` directory.

Produces two pages:
- ``site/index.html`` — ranker output (top-24 chart, model report, ranking CSVs).
- ``site/simulation.html`` — tournament simulator output (deterministic bracket,
  Monte Carlo probabilities, medals).

Both pages share a common stylesheet and link to each other via a top nav.
Run after ``worldcup-ranker rank`` and (optionally) ``worldcup-ranker simulate``.
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
RANKING_ASSETS = (
    "top_24_rankings.csv",
    "full_rankings.csv",
    "model_report.md",
    "top_24.png",
)
SIMULATION_ASSETS = (
    "simulation_report.md",
    "deterministic_bracket.md",
    "monte_carlo_probabilities.csv",
    "monte_carlo_medals.csv",
)


_SHARED_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem 4rem;
  color: #1f2328;
  background: #fff;
  line-height: 1.55;
}
h1, h2, h3 { line-height: 1.25; }
h1 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }
h2 { margin-top: 2rem; border-bottom: 1px solid #eaecef; padding-bottom: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-variant-numeric: tabular-nums; }
th, td { padding: 0.45rem 0.7rem; border-bottom: 1px solid #eaecef; }
th { background: #f6f8fa; text-align: left; }
td:nth-child(n+3), th:nth-child(n+3) { text-align: right; }
img { max-width: 100%; height: auto; display: block; margin: 1rem auto; }
code { background: #f6f8fa; padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.95em; }
.meta { color: #57606a; font-size: 0.9rem; }
.downloads {
  margin: 1rem 0;
  padding: 0.75rem 1rem;
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  border-radius: 8px;
}
.downloads a { margin-right: 1rem; }
nav.top {
  display: flex;
  gap: 1.25rem;
  margin-bottom: 1.5rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #d0d7de;
  font-size: 0.95rem;
}
nav.top a { text-decoration: none; }
nav.top a.current { font-weight: 600; }
@media (prefers-color-scheme: dark) {
  body { background: #0d1117; color: #c9d1d9; }
  h1 { border-color: #30363d; }
  h2 { border-color: #21262d; }
  nav.top { border-color: #30363d; }
  th { background: #161b22; }
  th, td { border-color: #21262d; }
  code, .downloads { background: #161b22; border-color: #30363d; }
  .meta { color: #8b949e; }
  a { color: #58a6ff; }
}
"""


def _page(title: str, nav_html: str, content_html: str, built_at: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_SHARED_CSS}</style>
</head>
<body>
{nav_html}
<p class="meta">Built {built_at}.</p>
{content_html}
</body>
</html>
"""


def _nav(current: str) -> str:
    def cls(name: str) -> str:
        return ' class="current"' if name == current else ""

    return (
        '<nav class="top">'
        f'<a href="index.html"{cls("ranking")}>Strength ranking</a>'
        f'<a href="simulation.html"{cls("simulation")}>Tournament simulation</a>'
        "</nav>"
    )


def _md_to_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def _copy_assets(assets: tuple[str, ...]) -> list[str]:
    """Copy assets that exist; return the names actually copied."""
    copied = []
    for name in assets:
        src = OUTPUTS / name
        if src.exists():
            shutil.copy2(src, SITE / name)
            copied.append(name)
    return copied


def build_ranking_page(built_at: str) -> bool:
    report_path = OUTPUTS / "model_report.md"
    if not report_path.exists():
        sys.stderr.write(f"Missing {report_path}; skipping ranking page.\n")
        return False

    _copy_assets(RANKING_ASSETS)

    body_html = _md_to_html(report_path)
    chart_html = (
        '<img src="top_24.png" alt="Top 24 strength ranking">'
        if (SITE / "top_24.png").exists()
        else ""
    )
    downloads = (
        '<div class="downloads"><strong>Downloads:</strong> '
        '<a href="top_24_rankings.csv">top_24_rankings.csv</a> '
        '<a href="full_rankings.csv">full_rankings.csv</a> '
        '<a href="model_report.md">model_report.md</a>'
        "</div>"
    )
    content = f"{downloads}{chart_html}{body_html}"
    html = _page("World Cup Strength Ranking", _nav("ranking"), content, built_at)
    (SITE / "index.html").write_text(html, encoding="utf-8")
    return True


def build_simulation_page(built_at: str) -> bool:
    sim_report = OUTPUTS / "simulation_report.md"
    if not sim_report.exists():
        sys.stderr.write(
            "No outputs/simulation_report.md; skipping simulation page.\n"
        )
        return False

    _copy_assets(SIMULATION_ASSETS)

    report_html = _md_to_html(sim_report)
    bracket_path = OUTPUTS / "deterministic_bracket.md"
    bracket_html = ""
    if bracket_path.exists():
        bracket_html = (
            '<hr><section id="deterministic-walk">'
            f"{_md_to_html(bracket_path)}"
            "</section>"
        )

    downloads = (
        '<div class="downloads"><strong>Downloads:</strong> '
        '<a href="simulation_report.md">simulation_report.md</a> '
        '<a href="deterministic_bracket.md">deterministic_bracket.md</a> '
        '<a href="monte_carlo_probabilities.csv">monte_carlo_probabilities.csv</a> '
        '<a href="monte_carlo_medals.csv">monte_carlo_medals.csv</a>'
        "</div>"
    )
    content = f"{downloads}{report_html}{bracket_html}"
    html = _page(
        "2026 World Cup Tournament Simulation",
        _nav("simulation"),
        content,
        built_at,
    )
    (SITE / "simulation.html").write_text(html, encoding="utf-8")
    return True


def main() -> int:
    if not OUTPUTS.exists():
        sys.stderr.write(
            "outputs/ does not exist. Run `worldcup-ranker rank` first.\n"
        )
        return 1

    SITE.mkdir(exist_ok=True)
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ranking_ok = build_ranking_page(built_at)
    sim_ok = build_simulation_page(built_at)

    if ranking_ok:
        print(f"Built {SITE / 'index.html'}")
    if sim_ok:
        print(f"Built {SITE / 'simulation.html'}")
    if not ranking_ok and not sim_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
