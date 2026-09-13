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
            "refundability": 0.0,
        },
        "balanced": {
            "preference": 0.30,
            "quality": 0.20,
            "location": 0.20,
            "convenience": 0.15,
            "price": 0.15,
            "refundability": 0.0,
        },
        "experience": {
            "preference": 0.40,
            "quality": 0.30,
            "location": 0.15,
            "convenience": 0.10,
            "price": 0.05,
            "refundability": 0.0,
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
            for source in disrupted:
                targets = np.flatnonzero(
                    observation["swap_mask"][1, source]
                )
                if len(targets):
                    category = int(np.argmax(inventory[source, :3]))
                    target = self._best_from_candidates(
                        observation,
                        [int(index) for index in targets],
                        category,
                    )
                    return self._action(
                        ActionType.REBOOK,
                        target,
                        int(source),
                    )

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
            required = (
                max(1, int(round(observation["trip_state"][13] * 4)))
                if category == 2
                else 1
            )
            if booked_categories.count(category) < required:
                target = self._best_target(
                    observation, select_action, category
                )
                if target is not None:
                    return self._action(select_action, target)

        if observation["trip_state"][12] > 0.5:
            return self._action(ActionType.FINISH)

        if observation["trip_state"][7] > 0.5:
            return self._action(ActionType.ADVANCE_TRIP)

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

        remaining_budget = float(observation["trip_state"][1])
        reserve = self._minimum_reserve(observation, category)
        affordable = [
            index
            for index in candidates
            if observation["inventory"][index, 3] + reserve
            <= remaining_budget + 1e-6
        ]
        if affordable:
            candidates = affordable
        return self._best_from_candidates(observation, candidates, category)

    def _best_from_candidates(
        self,
        observation: Dict[str, np.ndarray],
        candidates,
        category: int,
    ):
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
                + self.weights["refundability"] * row[13]
            )
            scores.append(score)
        return candidates[int(np.argmax(scores))]

    @staticmethod
    def _minimum_reserve(
        observation: Dict[str, np.ndarray], selecting_category: int
    ) -> float:
        inventory = observation["inventory"]
        visible = observation["inventory_mask"]
        booked = set(np.flatnonzero(observation["booking_mask"]).tolist())
        booked_counts = [0, 0, 0]
        for index in booked:
            booked_counts[int(np.argmax(inventory[index, :3]))] += 1
        required = [
            1,
            1,
            max(1, int(round(observation["trip_state"][13] * 4))),
        ]
        required[selecting_category] = max(
            0,
            required[selecting_category] - 1,
        )

        reserve = 0.0
        for category in range(3):
            needed = max(0, required[category] - booked_counts[category])
            prices = sorted(
                float(inventory[index, 3])
                for index in range(len(inventory))
                if visible[index]
                and index not in booked
                and inventory[index, 14] > 0.5
                and int(np.argmax(inventory[index, :3])) == category
            )
            reserve += sum(prices[:needed])
        return reserve

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

        valid_pairs = []
        for source in activity_bookings:
            for target in np.flatnonzero(
                observation["swap_mask"][0, source]
            ):
                valid_pairs.append((source, int(target)))
        if not valid_pairs:
            return None
        source, target = max(
            valid_pairs,
            key=lambda pair: preference_score(pair[1])
            - preference_score(pair[0]),
        )
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
