from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Union

import yaml


@dataclass(frozen=True)
class RewardWeights:
    preference: float = 3.0
    coherence: float = 2.0
    budget: float = 1.5
    quality: float = 1.5
    convenience: float = 1.0
    recovery: float = 2.0
    violations: float = 2.5
    step_cost: float = 0.01
    invalid_action: float = 0.25
    incomplete_finish: float = 3.0
    useful_booking: float = 0.05


@dataclass(frozen=True)
class EnvironmentConfig:
    max_inventory: int = 32
    max_steps: int = 40
    min_activities: int = 2
    discount: float = 0.99
    client_patience: int = 4
    acceptance_threshold: float = 0.58
    cancellation_fee_rate: float = 0.10
    disruption_probability: float = 0.0
    difficulty: int = 1
    reward: RewardWeights = field(default_factory=RewardWeights)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EnvironmentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        reward_raw = raw.pop("reward", {})
        return cls(
            **_known_fields(cls, raw),
            reward=RewardWeights(**_known_fields(RewardWeights, reward_raw)),
        )


def _known_fields(model: Any, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {item.name for item in fields(model)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {model.__name__} settings: {sorted(unknown)}")
    return values
