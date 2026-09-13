from __future__ import annotations

from typing import Dict

import numpy as np

from travel_env.actions import ActionType


class HeuristicPolicy:
    PROFILES = {
        "budget": {
            "preference": 0.15,
            "quality": 0.10,
            "location": 0.10,
            "convenience": 0.10,
            "price": 0.55,
        },
        "balanced": {
            "preference": 0.30,
            "quality": 0.20,
            "location": 0.20,
            "convenience": 0.15,
            "price": 0.15,
        },
        "experience": {
            "preference": 0.40,
            "quality": 0.30,
            "location": 0.15,
            "convenience": 0.10,
            "price": 0.05,
        },
    }

    def __init__(self, profile: str = "balanced"):
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown heuristic profile: {profile}")
        self.weights = self.PROFILES[profile]

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, int]:
        action_mask = observation["action_mask"]
        booking_mask = observation["booking_mask"]
        inventory = observation["inventory"]

        disrupted = np.flatnonzero(
            observation["disruption_mask"] * booking_mask
        )
        if len(disrupted) and action_mask[ActionType.REBOOK]:
            source = int(disrupted[0])
            category = int(np.argmax(inventory[source, :3]))
            target = self._best_target(
                observation, ActionType.REBOOK, category
            )
            if target is not None:
                return self._action(ActionType.REBOOK, target, source)

        visible_categories = {
            int(np.argmax(row[:3]))
            for row, visible in zip(
                inventory, observation["inventory_mask"]
            )
            if visible
        }
        for category, search_action in enumerate(
            (
                ActionType.SEARCH_FLIGHTS,
                ActionType.SEARCH_HOTELS,
                ActionType.SEARCH_ACTIVITIES,
            )
        ):
            if category not in visible_categories:
                return self._action(search_action)

        booked_categories = [
            int(np.argmax(inventory[index, :3]))
            for index in np.flatnonzero(booking_mask)
        ]
        for category, select_action in enumerate(
            (
                ActionType.SELECT_FLIGHT,
                ActionType.SELECT_HOTEL,
                ActionType.SELECT_ACTIVITY,
            )
        ):
            required = 2 if category == 2 else 1
            if booked_categories.count(category) < required:
                target = self._best_target(
                    observation, select_action, category
                )
                if target is not None:
                    return self._action(select_action, target)

        if observation["trip_state"][7] > 0.5:
            return self._action(ActionType.FINISH)

        if observation["trip_state"][9] > 0.5:
            replacement = self._revision_action(observation)
            if replacement is not None:
                return replacement

        if action_mask[ActionType.PROPOSE_ITINERARY]:
            return self._action(ActionType.PROPOSE_ITINERARY)

        if action_mask[ActionType.MESSAGE_CLIENT]:
            return self._action(ActionType.MESSAGE_CLIENT)

        return self._action(ActionType.SEARCH_ACTIVITIES)

    def _best_target(
        self,
        observation: Dict[str, np.ndarray],
        action_type: ActionType,
        category: int,
    ):
        candidates = np.flatnonzero(
            observation["target_mask"][int(action_type)]
        )
        booked = set(np.flatnonzero(observation["booking_mask"]).tolist())
        candidates = [
            int(index)
            for index in candidates
            if index not in booked
            and int(np.argmax(observation["inventory"][index, :3])) == category
        ]
        if not candidates:
            return None

        preferences = observation["preferences"]
        scores = []
        for index in candidates:
            row = observation["inventory"][index]
            price = row[3]
            quality = row[4]
            location = row[5]
            convenience = row[6]
            theme_fit = float(np.dot(preferences, row[7:11]))
            score = (
                self.weights["preference"] * theme_fit
                + self.weights["quality"] * quality
                + self.weights["location"] * location
                + self.weights["convenience"] * convenience
                + self.weights["price"] * (1.0 - price)
            )
            scores.append(score)
        return candidates[int(np.argmax(scores))]

    def _revision_action(self, observation: Dict[str, np.ndarray]):
        inventory = observation["inventory"]
        booked = np.flatnonzero(observation["booking_mask"])
        preferences = observation["preferences"]
        activity_bookings = [
            int(index)
            for index in booked
            if int(np.argmax(inventory[index, :3])) == 2
        ]
        if not activity_bookings:
            return None

        def preference_score(index: int) -> float:
            row = inventory[index]
            return float(
                0.60 * np.dot(preferences, row[7:11])
                + 0.15 * row[4]
                + 0.10 * row[5]
                + 0.10 * row[6]
                + 0.05 * (1.0 - row[3])
            )

        source = min(activity_bookings, key=preference_score)
        candidates = [
            int(index)
            for index in np.flatnonzero(
                observation["target_mask"][ActionType.SWAP_BOOKING]
            )
            if int(np.argmax(inventory[index, :3])) == 2
            and index not in booked
        ]
        if not candidates:
            return None
        target = max(candidates, key=preference_score)
        return self._action(ActionType.SWAP_BOOKING, target, source)

    @staticmethod
    def _action(
        action_type: ActionType, target: int = 0, source: int = 0
    ) -> Dict[str, int]:
        return {
            "action_type": int(action_type),
            "target_index": target,
            "source_index": source,
        }
