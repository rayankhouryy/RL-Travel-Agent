from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set

import numpy as np


THEMES = ("food", "history", "nightlife", "nature")


class InventoryCategory(IntEnum):
    FLIGHT = 0
    HOTEL = 1
    ACTIVITY = 2


@dataclass
class ClientPersona:
    preference_weights: np.ndarray
    budget_sensitivity: float
    quality_preference: float
    location_preference: float
    convenience_preference: float
    expected_budget_usage: float
    budget_flexibility: float
    change_aversion: float
    disruption_tolerance: float
    communication_style: int


@dataclass
class TripRequest:
    destination: int
    party_size: int
    duration_days: int
    budget: float
    stated_preferences: np.ndarray


@dataclass
class InventoryItem:
    index: int
    item_id: str
    category: InventoryCategory
    price: float
    quality: float
    location: float
    convenience: float
    theme_vector: np.ndarray
    start_hour: float
    end_hour: float
    refundable: bool
    available: bool = True


@dataclass
class Disruption:
    target_index: int
    kind: str
    triggered_step: int
    affected_indices: Set[int] = field(default_factory=set)
    resolved: bool = False


@dataclass
class TravelState:
    persona: ClientPersona
    request: TripRequest
    inventory: List[InventoryItem]
    visible: np.ndarray
    difficulty: int = 1
    required_activities: int = 2
    disruption_probability: float = 0.0
    max_disruptions: int = 0
    step_limit: int = 40
    booked: Set[int] = field(default_factory=set)
    spent: float = 0.0
    step_count: int = 0
    search_count: int = 0
    proposal_count: int = 0
    rebooking_count: int = 0
    patience_remaining: int = 4
    feedback_revealed: np.ndarray = field(
        default_factory=lambda: np.zeros(len(THEMES), dtype=np.float32)
    )
    awaiting_revision: bool = False
    client_accepted: bool = False
    current_day: int = 0
    trip_started: bool = False
    trip_completed: bool = False
    done: bool = False
    disruptions: List[Disruption] = field(default_factory=list)
    invalid_actions: int = 0

    def items_by_category(self, category: InventoryCategory) -> List[InventoryItem]:
        return [self.inventory[index] for index in self.booked
                if self.inventory[index].category == category]

    def has_complete_itinerary(self) -> bool:
        return (
            len(self.items_by_category(InventoryCategory.FLIGHT)) == 1
            and len(self.items_by_category(InventoryCategory.HOTEL)) == 1
            and len(self.items_by_category(InventoryCategory.ACTIVITY))
            >= self.required_activities
            and all(self.inventory[index].available for index in self.booked)
        )

    def hard_budget(self) -> float:
        return self.request.budget * (1.0 + self.persona.budget_flexibility)

    def booking_for(self, category: InventoryCategory) -> Optional[InventoryItem]:
        matches = self.items_by_category(category)
        return matches[0] if matches else None

    def activity_conflicts(
        self, candidate: InventoryItem, ignore: Optional[int] = None
    ) -> bool:
        for index in self.booked:
            if index == ignore:
                continue
            item = self.inventory[index]
            if item.category != InventoryCategory.ACTIVITY:
                continue
            if candidate.start_hour < item.end_hour and item.start_hour < candidate.end_hour:
                return True
        return False

    def active_disruption_targets(self) -> Set[int]:
        targets: Set[int] = set()
        for disruption in self.disruptions:
            if disruption.resolved:
                continue
            targets.add(disruption.target_index)
            targets.update(disruption.affected_indices)
        return targets

    def booked_items(self) -> List[InventoryItem]:
        return [self.inventory[index] for index in sorted(self.booked)]

    def booking_counts(self) -> Dict[InventoryCategory, int]:
        return {
            category: len(self.items_by_category(category))
            for category in InventoryCategory
        }
