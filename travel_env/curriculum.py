from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DifficultyProfile:
    level: int
    required_activities: int
    preference_dimensions_visible: int
    patience: int
    budget_low: float
    budget_high: float
    scarcity_multiplier: float
    disruption_multiplier: float
    max_disruptions: int
    step_limit: int


DIFFICULTY_PROFILES: Dict[int, DifficultyProfile] = {
    1: DifficultyProfile(
        level=1,
        required_activities=1,
        preference_dimensions_visible=3,
        patience=5,
        budget_low=0.98,
        budget_high=1.28,
        scarcity_multiplier=1.00,
        disruption_multiplier=0.00,
        max_disruptions=0,
        step_limit=35,
    ),
    2: DifficultyProfile(
        level=2,
        required_activities=2,
        preference_dimensions_visible=2,
        patience=4,
        budget_low=0.90,
        budget_high=1.18,
        scarcity_multiplier=1.08,
        disruption_multiplier=1.00,
        max_disruptions=1,
        step_limit=45,
    ),
    3: DifficultyProfile(
        level=3,
        required_activities=3,
        preference_dimensions_visible=2,
        patience=3,
        budget_low=0.84,
        budget_high=1.10,
        scarcity_multiplier=1.18,
        disruption_multiplier=2.00,
        max_disruptions=2,
        step_limit=60,
    ),
    4: DifficultyProfile(
        level=4,
        required_activities=4,
        preference_dimensions_visible=1,
        patience=3,
        budget_low=0.80,
        budget_high=1.04,
        scarcity_multiplier=1.30,
        disruption_multiplier=3.50,
        max_disruptions=3,
        step_limit=75,
    ),
}


def difficulty_profile(level: int) -> DifficultyProfile:
    try:
        return DIFFICULTY_PROFILES[level]
    except KeyError as exc:
        raise ValueError(
            f"difficulty must be one of {sorted(DIFFICULTY_PROFILES)}, got {level}"
        ) from exc
