from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from travel_env.config import EnvironmentConfig
from travel_env.models import InventoryCategory, TravelState


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: Dict[str, float]


class RewardModel:
    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def potential(self, state: TravelState) -> float:
        counts = state.booking_counts()
        flight = min(1.0, float(counts[InventoryCategory.FLIGHT]))
        hotel = min(1.0, float(counts[InventoryCategory.HOTEL]))
        activities = min(
            1.0,
            counts[InventoryCategory.ACTIVITY]
            / max(state.required_activities, 1),
        )
        coverage = (flight + hotel + activities) / 3.0
        return coverage

    def transition(
        self,
        previous_potential: float,
        state: TravelState,
        *,
        valid: bool,
        finishing: bool,
        terminal: bool,
    ) -> RewardResult:
        weights = self.config.reward
        next_potential = 0.0 if terminal else self.potential(state)
        shaping = weights.shaping_scale * (
            self.config.discount * next_potential - previous_potential
        )
        components = {
            "step_cost": -weights.step_cost,
            "potential_shaping": shaping,
            "invalid_action": -weights.invalid_action if not valid else 0.0,
        }

        if finishing:
            if not valid:
                components["terminal_failure"] = -weights.incomplete_finish
            elif not state.has_complete_itinerary():
                components["terminal_failure"] = -weights.incomplete_finish
            elif not state.client_accepted:
                components["terminal_failure"] = (
                    -0.5 * weights.incomplete_finish
                )
            else:
                components.update(self.terminal_components(state))

        return RewardResult(
            total=float(sum(components.values())),
            components={key: float(value) for key, value in components.items()},
        )

    def terminal_components(self, state: TravelState) -> Dict[str, float]:
        weights = self.config.reward
        return {
            "client_utility": weights.utility * self.latent_utility(state),
        }

    def itinerary_metrics(self, state: TravelState) -> Dict[str, float]:
        items = state.booked_items()
        activities = [
            item for item in items if item.category == InventoryCategory.ACTIVITY
        ]
        preference = (
            float(
                np.mean(
                    [
                        np.dot(
                            state.persona.preference_weights,
                            item.theme_vector,
                        )
                        for item in activities
                    ]
                )
            )
            if activities
            else 0.0
        )
        quality = float(np.mean([item.quality for item in items])) if items else 0.0
        convenience = (
            float(np.mean([item.convenience for item in items])) if items else 0.0
        )
        target_spend = (
            state.request.budget * state.persona.expected_budget_usage
        )
        budget = float(
            np.exp(
                -max(state.spent - target_spend, 0.0)
                / max(target_spend, 1.0)
            )
        )
        coherence = 1.0 if not self._has_overlap(activities) else 0.0
        active_failures = len(state.active_disruption_targets() & state.booked)
        recovery = 1.0 if state.disruptions and active_failures == 0 else 0.0
        violations = float(active_failures)

        return {
            "preference_match": preference,
            "coherence": coherence,
            "budget_score": budget,
            "quality": quality,
            "convenience": convenience,
            "recovery": recovery,
            "violations": violations,
        }

    def latent_utility(self, state: TravelState) -> float:
        items = state.booked_items()
        if not items:
            return 0.0
        activities = [
            item for item in items if item.category == InventoryCategory.ACTIVITY
        ]
        theme_fit = (
            float(
                np.mean(
                    [
                        np.dot(state.persona.preference_weights, item.theme_vector)
                        for item in activities
                    ]
                )
            )
            if activities
            else 0.0
        )
        qualities = [item.quality for item in items]
        quality = float(np.mean(qualities))
        quality_shortfall = max(
            state.persona.quality_floor - min(qualities),
            0.0,
        )
        quality_fit = quality * float(np.exp(-3.0 * quality_shortfall))
        location = float(np.mean([item.location for item in items]))
        convenience = float(np.mean([item.convenience for item in items]))
        target = state.request.budget * state.persona.expected_budget_usage
        budget_fit = float(
            np.exp(-max(state.spent - target, 0.0) / max(target, 1.0))
        )
        pace = len(activities) / max(state.request.duration_days, 1)
        pace_fit = float(
            np.exp(-abs(pace - state.persona.preferred_pace) / 0.65)
        )
        values = np.array(
            [
                min(1.0, 2.0 * theme_fit),
                quality_fit,
                location,
                convenience,
                budget_fit,
                pace_fit,
            ],
            dtype=np.float64,
        )
        importance = np.array(
            [
                1.5,
                state.persona.quality_preference,
                state.persona.location_preference,
                state.persona.convenience_preference,
                state.persona.budget_sensitivity,
                1.0,
            ],
            dtype=np.float64,
        )
        return float(
            np.clip(
                np.dot(values, importance) / max(importance.sum(), 1e-8),
                0.0,
                1.0,
            )
        )

    def realized_satisfaction(self, state: TravelState) -> float:
        if not state.has_complete_itinerary():
            return 0.0
        items = state.booked_items()
        activities = [
            item for item in items if item.category == InventoryCategory.ACTIVITY
        ]
        theme_fit = (
            float(
                np.mean(
                    [
                        np.dot(
                            state.persona.preference_weights,
                            item.theme_vector,
                        )
                        for item in activities
                    ]
                )
            )
            if activities
            else 0.0
        )
        qualities = [item.quality for item in items]
        quality = float(np.mean(qualities)) if qualities else 0.0
        if qualities and min(qualities) < state.persona.quality_floor:
            shortfall = state.persona.quality_floor - min(qualities)
            quality *= max(0.15, 1.0 - 2.4 * shortfall)
        location = float(np.mean([item.location for item in items]))
        convenience = float(np.mean([item.convenience for item in items]))
        pace = len(activities) / max(state.request.duration_days, 1)
        pace_fit = float(
            np.exp(-abs(pace - state.persona.preferred_pace) / 0.8)
        )
        target = state.request.budget * state.persona.expected_budget_usage
        budget_fit = float(
            np.exp(-max(state.spent - target, 0.0) / max(target, 1.0))
        )
        outcome = (
            0.30 * min(1.0, 2.0 * theme_fit)
            + 0.22 * quality
            + 0.13 * location
            + 0.10 * convenience
            + 0.15 * pace_fit
            + 0.10 * budget_fit
        )
        robustness = float(
            np.mean([1.0 if item.refundable else 0.35 for item in items])
        )
        unresolved = len(state.active_disruption_targets() & state.booked)
        disruption_factor = max(0.0, 1.0 - 0.3 * unresolved)
        sunk_loss_factor = max(
            0.0,
            1.0 - 1.5 * state.sunk_cost / max(state.hard_budget(), 1.0),
        )
        return float(
            np.clip(
                0.85 * outcome
                + 0.15
                * robustness
                * state.persona.disruption_sensitivity,
                0.0,
                1.0,
            )
            * disruption_factor
            * sunk_loss_factor
        )

    @staticmethod
    def _has_overlap(activities: list) -> bool:
        ordered = sorted(activities, key=lambda item: item.start_hour)
        return any(
            current.end_hour > following.start_hour
            for current, following in zip(ordered, ordered[1:])
        )
