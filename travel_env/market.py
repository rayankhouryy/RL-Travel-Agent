from __future__ import annotations

from collections import Counter

import numpy as np

from travel_env.config import EnvironmentConfig
from travel_env.models import InventoryCategory, TravelState


class MarketEngine:
    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def advance(
        self, state: TravelState, rng: np.random.Generator
    ) -> None:
        state.market_steps += 1
        availability = Counter(
            item.category
            for item in state.inventory
            if item.available and item.index not in state.booked
        )
        booked_counts = state.booking_counts()

        for item in state.inventory:
            if item.index in state.booked or not item.available:
                continue

            drift = self.config.price_drift_per_step * (
                0.75 + 0.50 * rng.random()
            )
            item.price = round(item.price * (1.0 + drift), 2)

            if not state.visible[item.index]:
                continue
            probability = (
                self.config.depletion_probability
                * max(1.0, state.difficulty / 2.0)
            )
            if rng.random() >= probability:
                continue

            required_remaining = self._required_remaining(
                state,
                item.category,
                booked_counts,
            )
            protected = max(1, required_remaining)
            if availability[item.category] <= protected:
                continue

            item.available = False
            availability[item.category] -= 1
            state.depleted_inventory += 1

    @staticmethod
    def _required_remaining(
        state: TravelState,
        category: InventoryCategory,
        booked_counts: Counter,
    ) -> int:
        required = {
            InventoryCategory.FLIGHT: 1,
            InventoryCategory.HOTEL: 1,
            InventoryCategory.ACTIVITY: state.required_activities,
        }[category]
        return max(0, required - booked_counts[category])
