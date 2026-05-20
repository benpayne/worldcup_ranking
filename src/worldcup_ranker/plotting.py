"""Optional chart of the top-N teams."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402

from .ranking import RankingResult  # noqa: E402


def plot_top_n(result: RankingResult, output_dir: str | Path, top_n: int = 24) -> Path:
    """Render a horizontal bar chart of the top-N teams to ``top_<N>.png``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"top_{top_n}.png"

    head = result.rankings.head(top_n).iloc[::-1]  # reverse so #1 is at top
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh(head["team"], head["final_score"], color="#1f6feb")
    ax.set_xlabel("Final score (0-100)")
    ax.set_title(f"Top {top_n} - pre-tournament strength ranking (cutoff {result.cutoff.date()})")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
