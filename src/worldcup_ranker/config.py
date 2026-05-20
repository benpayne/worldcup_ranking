"""Configuration loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class TournamentConfig(BaseModel):
    start_date: str
    form_window_days: int = Field(730, ge=30)
    min_matches: int = Field(5, ge=1)
    drop_below_min_matches: bool = True


class Weights(BaseModel):
    elo: float = 0.50
    recent_form: float = 0.25
    goal_performance: float = 0.15
    squad_strength: float = 0.10

    @field_validator("elo", "recent_form", "goal_performance", "squad_strength")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weights must be non-negative")
        return v

    def total(self) -> float:
        return self.elo + self.recent_form + self.goal_performance + self.squad_strength


class EloConfig(BaseModel):
    initial_rating: float = 1500.0
    home_advantage: float = 65.0
    k_base: float = 30.0
    importance: dict[str, float] = Field(default_factory=dict)
    default_importance: float = 1.0
    goal_diff_multiplier: bool = True


class RecentFormConfig(BaseModel):
    half_life_days: float = 365.0
    opponent_adjusted: bool = True


class GoalPerformanceConfig(BaseModel):
    goal_diff_cap: int = 5
    attack_weight: float = 0.5
    defense_weight: float = 0.5


class SquadStrengthConfig(BaseModel):
    csv_path: Optional[str] = None


class DataConfig(BaseModel):
    results_csv: str = "data/raw/results.csv"
    qualified_teams_csv: Optional[str] = None
    team_aliases: dict[str, str] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    top_n: int = 24
    directory: str = "outputs"
    generate_chart: bool = True


class AppConfig(BaseModel):
    tournament: TournamentConfig
    weights: Weights = Weights()
    elo: EloConfig = EloConfig()
    recent_form: RecentFormConfig = RecentFormConfig()
    goal_performance: GoalPerformanceConfig = GoalPerformanceConfig()
    squad_strength: SquadStrengthConfig = SquadStrengthConfig()
    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config and validate via pydantic."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig(**raw)
