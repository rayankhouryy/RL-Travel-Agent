from __future__ import annotations

from typing import List, Optional

import numpy as np

from travel_env.config import EnvironmentConfig
from travel_env.models import (
    THEMES,
    ClientPersona,
    InventoryCategory,
    InventoryItem,
    TravelState,
    TripRequest,
)


class TravelWorldGenerator:
    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def generate_episode(
        self, rng: np.random.Generator, difficulty: Optional[int] = None
    ) -> TravelState:
        difficulty = difficulty or self.config.difficulty
        persona = self._persona(rng, difficulty)
        request = self._request(rng, persona, difficulty)
        inventory = self._inventory(rng, request, difficulty)
        return TravelState(
            persona=persona,
            request=request,
            inventory=inventory,
            visible=np.zeros(self.config.max_inventory, dtype=np.int8),
            patience_remaining=max(1, self.config.client_patience - difficulty + 1),
        )

    def _persona(
        self, rng: np.random.Generator, difficulty: int
    ) -> ClientPersona:
        preferences = rng.dirichlet(np.full(len(THEMES), 0.8 + 0.15 * difficulty))
        return ClientPersona(
            preference_weights=preferences.astype(np.float32),
            budget_sensitivity=float(rng.beta(2.0, 2.0)),
            quality_preference=float(rng.beta(2.2, 1.8)),
            location_preference=float(rng.beta(2.0, 2.0)),
            convenience_preference=float(rng.beta(2.0, 2.0)),
            expected_budget_usage=float(rng.uniform(0.72, 0.96)),
            budget_flexibility=float(rng.uniform(0.02, 0.18)),
            change_aversion=float(rng.beta(2.0, 3.0)),
            disruption_tolerance=float(rng.beta(2.0, 2.0)),
            communication_style=int(rng.integers(0, 4)),
        )

    def _request(
        self,
        rng: np.random.Generator,
        persona: ClientPersona,
        difficulty: int,
    ) -> TripRequest:
        duration = int(rng.integers(4, 9))
        party_size = int(rng.integers(1, 5))
        base_budget = 950 + 320 * duration + 430 * party_size
        budget = float(base_budget * rng.uniform(0.82, 1.25 - 0.04 * difficulty))

        stated = np.zeros(len(THEMES), dtype=np.float32)
        visible_count = max(1, 3 - difficulty // 2)
        visible_dimensions = np.argsort(persona.preference_weights)[-visible_count:]
        noise = rng.normal(0.0, 0.06, visible_count)
        stated[visible_dimensions] = np.clip(
            persona.preference_weights[visible_dimensions] + noise, 0.0, 1.0
        )
        return TripRequest(
            destination=int(rng.integers(0, 8)),
            party_size=party_size,
            duration_days=duration,
            budget=round(budget, 2),
            stated_preferences=stated,
        )

    def _inventory(
        self,
        rng: np.random.Generator,
        request: TripRequest,
        difficulty: int,
    ) -> List[InventoryItem]:
        destination_base = float(rng.uniform(0.85, 1.35))
        season = float(rng.uniform(0.85, 1.45))
        scarcity = 1.0 + 0.08 * difficulty
        items: List[InventoryItem] = []

        for _ in range(6):
            quality = float(rng.beta(2.4, 1.9))
            convenience = float(rng.beta(2.0, 2.0))
            price = (
                190
                * request.party_size
                * destination_base
                * season
                * scarcity
                * (0.72 + 0.75 * quality)
                * (0.75 + 0.65 * convenience)
                * rng.lognormal(0.0, 0.13)
            )
            departure = float(rng.uniform(4, 18))
            duration = float(rng.uniform(5, 14))
            items.append(
                self._item(
                    items,
                    InventoryCategory.FLIGHT,
                    price,
                    quality,
                    location=1.0,
                    convenience=convenience,
                    theme=np.zeros(len(THEMES), dtype=np.float32),
                    start=departure,
                    end=departure + duration,
                    refundable=bool(rng.random() < 0.35),
                )
            )

        for _ in range(8):
            quality = float(rng.beta(2.2, 1.8))
            location = float(rng.beta(2.0, 2.0))
            popularity = float(rng.beta(2.0, 2.0))
            nightly = (
                95
                * destination_base
                * season
                * scarcity
                * (0.62 + 0.95 * quality)
                * (0.72 + 0.75 * location)
                * rng.lognormal(0.0, 0.18)
            )
            items.append(
                self._item(
                    items,
                    InventoryCategory.HOTEL,
                    nightly * request.duration_days,
                    quality,
                    location=location,
                    convenience=0.65 * location + 0.35 * quality,
                    theme=rng.dirichlet(np.full(len(THEMES), 1.5)).astype(np.float32),
                    start=0.0,
                    end=float(request.duration_days * 24),
                    refundable=bool(rng.random() < 0.55),
                    available=bool(rng.random() > 0.06 + 0.12 * popularity),
                )
            )

        while len(items) < self.config.max_inventory:
            quality = float(rng.beta(2.0, 2.0))
            popularity = float(rng.beta(1.8, 2.2))
            day = int(rng.integers(0, request.duration_days))
            start = float(day * 24 + rng.integers(9, 19))
            duration = float(rng.choice([2, 3, 4]))
            theme = rng.dirichlet(np.full(len(THEMES), 0.55)).astype(np.float32)
            price = (
                (32 + 105 * quality + 45 * popularity)
                * request.party_size
                * destination_base
                * season
                * rng.lognormal(0.0, 0.16)
            )
            items.append(
                self._item(
                    items,
                    InventoryCategory.ACTIVITY,
                    price,
                    quality,
                    location=float(rng.beta(2.0, 2.0)),
                    convenience=float(rng.beta(2.0, 2.0)),
                    theme=theme,
                    start=start,
                    end=start + duration,
                    refundable=bool(rng.random() < 0.45),
                    available=bool(rng.random() > 0.04 + 0.10 * popularity),
                )
            )
        return items

    @staticmethod
    def _item(
        items: List[InventoryItem],
        category: InventoryCategory,
        price: float,
        quality: float,
        location: float,
        convenience: float,
        theme: np.ndarray,
        start: float,
        end: float,
        refundable: bool,
        available: bool = True,
    ) -> InventoryItem:
        index = len(items)
        prefix = ("flight", "hotel", "activity")[int(category)]
        return InventoryItem(
            index=index,
            item_id=f"{prefix}_{index:03d}",
            category=category,
            price=round(float(price), 2),
            quality=float(np.clip(quality, 0.0, 1.0)),
            location=float(np.clip(location, 0.0, 1.0)),
            convenience=float(np.clip(convenience, 0.0, 1.0)),
            theme_vector=theme,
            start_hour=start,
            end_hour=end,
            refundable=refundable,
            available=available,
        )
