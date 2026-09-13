from __future__ import annotations

import numpy as np
from typing import Optional

from travel_env.config import EnvironmentConfig
from travel_env.models import Disruption, InventoryCategory, TravelState


class DisruptionEngine:
    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def maybe_trigger(
        self, state: TravelState, rng: np.random.Generator
    ) -> Optional[Disruption]:
        if not state.booked or rng.random() >= self.config.disruption_probability:
            return None
        candidates = [
            index
            for index in state.booked
            if state.inventory[index].available
        ]
        if not candidates:
            return None
        target = int(rng.choice(candidates))
        target_item = state.inventory[target]
        target_item.available = False
        affected = set()
        if target_item.category == InventoryCategory.FLIGHT:
            affected = {
                index
                for index in state.booked
                if state.inventory[index].category == InventoryCategory.ACTIVITY
                and state.inventory[index].start_hour < target_item.end_hour + 12.0
            }
            for index in affected:
                state.inventory[index].available = False

        kind = {
            InventoryCategory.FLIGHT: "flight_cancellation",
            InventoryCategory.HOTEL: "hotel_overbooking",
            InventoryCategory.ACTIVITY: "weather_closure",
        }[target_item.category]
        disruption = Disruption(
            target_index=target,
            kind=kind,
            triggered_step=state.step_count,
            affected_indices=affected,
        )
        state.disruptions.append(disruption)
        return disruption
