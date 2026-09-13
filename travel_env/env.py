from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from travel_env.actions import ActionType, TARGETED_ACTIONS, decode_action
from travel_env.config import EnvironmentConfig
from travel_env.disruptions import DisruptionEngine
from travel_env.generator import TravelWorldGenerator
from travel_env.models import THEMES, InventoryCategory, InventoryItem, TravelState
from travel_env.reward import RewardModel


class TravelAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.generator = TravelWorldGenerator(self.config)
        self.reward_model = RewardModel(self.config)
        self.disruption_engine = DisruptionEngine(self.config)
        self.state: Optional[TravelState] = None

        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(len(ActionType)),
                "source_index": spaces.Discrete(self.config.max_inventory),
                "target_index": spaces.Discrete(self.config.max_inventory),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "request": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
                "preferences": spaces.Box(
                    0.0, 1.0, shape=(len(THEMES),), dtype=np.float32
                ),
                "trip_state": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
                "inventory": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(self.config.max_inventory, 15),
                    dtype=np.float32,
                ),
                "inventory_mask": spaces.MultiBinary(self.config.max_inventory),
                "booking_mask": spaces.MultiBinary(self.config.max_inventory),
                "action_mask": spaces.MultiBinary(len(ActionType)),
                "target_mask": spaces.MultiBinary(
                    (len(ActionType), self.config.max_inventory)
                ),
                "disruption_mask": spaces.MultiBinary(self.config.max_inventory),
            }
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        difficulty = (options or {}).get("difficulty", self.config.difficulty)
        self.state = self.generator.generate_episode(self.np_random, difficulty)
        return self._observation(), self._info()

    def step(
        self, action: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        state = self._require_state()
        if state.done:
            raise RuntimeError("step() called after episode termination; call reset()")

        previous_potential = self.reward_model.potential(state)
        state.step_count += 1
        valid, useful_booking, finishing, reason = self._apply_action(action)

        if not valid:
            state.invalid_actions += 1

        disruption = None
        disruption_actions = {
            ActionType.SELECT_FLIGHT,
            ActionType.SELECT_HOTEL,
            ActionType.SELECT_ACTIVITY,
            ActionType.SWAP_BOOKING,
            ActionType.REBOOK,
        }
        try:
            decoded_action = ActionType(int(action["action_type"]))
        except (KeyError, TypeError, ValueError):
            decoded_action = None
        if valid and decoded_action in disruption_actions:
            disruption = self.disruption_engine.maybe_trigger(
                state, self.np_random
            )

        reward = self.reward_model.transition(
            previous_potential,
            state,
            valid=valid,
            useful_booking=useful_booking,
            finishing=finishing,
        )
        terminated = state.done
        truncated = state.step_count >= self.config.max_steps and not terminated
        info = self._info()
        info.update(
            {
                "action_valid": valid,
                "action_result": reason,
                "reward_components": reward.components,
                "triggered_disruption": (
                    disruption.target_index if disruption else None
                ),
            }
        )
        return self._observation(), reward.total, terminated, truncated, info

    def _apply_action(
        self, raw_action: Dict[str, int]
    ) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        try:
            action, target, source = decode_action(raw_action)
        except ValueError as exc:
            return False, False, False, str(exc)

        if action in TARGETED_ACTIONS and (
            not 0 <= target < len(state.inventory)
            or not 0 <= source < len(state.inventory)
        ):
            return False, False, False, "target_out_of_range"

        if action == ActionType.SEARCH_FLIGHTS:
            return self._search(InventoryCategory.FLIGHT)
        if action == ActionType.SEARCH_HOTELS:
            return self._search(InventoryCategory.HOTEL)
        if action == ActionType.SEARCH_ACTIVITIES:
            return self._search(InventoryCategory.ACTIVITY)
        if action == ActionType.SELECT_FLIGHT:
            return self._select(target, InventoryCategory.FLIGHT)
        if action == ActionType.SELECT_HOTEL:
            return self._select(target, InventoryCategory.HOTEL)
        if action == ActionType.SELECT_ACTIVITY:
            return self._select(target, InventoryCategory.ACTIVITY)
        if action == ActionType.REMOVE_BOOKING:
            return self._remove(target)
        if action == ActionType.SWAP_BOOKING:
            return self._swap(source, target, require_disrupted=False)
        if action == ActionType.REBOOK:
            return self._swap(source, target, require_disrupted=True)
        if action == ActionType.PROPOSE_ITINERARY:
            return self._propose()
        if action == ActionType.MESSAGE_CLIENT:
            return self._message()
        if action == ActionType.FINISH:
            return self._finish()
        return False, False, False, "unsupported_action"

    def _search(
        self, category: InventoryCategory
    ) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        state.search_count += 1
        for item in state.inventory:
            if item.category == category:
                state.visible[item.index] = 1
        return True, False, False, f"revealed_{category.name.lower()}"

    def _select(
        self, target: int, category: InventoryCategory
    ) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        item = state.inventory[target]
        if not state.visible[target]:
            return False, False, False, "inventory_not_visible"
        if item.category != category:
            return False, False, False, "wrong_inventory_category"
        if not item.available:
            return False, False, False, "inventory_unavailable"
        if target in state.booked:
            return False, False, False, "already_booked"
        if category in (InventoryCategory.FLIGHT, InventoryCategory.HOTEL):
            if state.booking_for(category) is not None:
                return False, False, False, "use_swap_for_existing_booking"
        if (
            category == InventoryCategory.ACTIVITY
            and state.activity_conflicts(item)
        ):
            return False, False, False, "schedule_conflict"
        if state.spent + item.price > state.hard_budget():
            return False, False, False, "hard_budget_exceeded"

        state.booked.add(target)
        state.spent += item.price
        state.awaiting_revision = False
        state.client_accepted = False
        return True, True, False, "booking_created"

    def _remove(self, target: int) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        if target not in state.booked:
            return False, False, False, "booking_not_found"
        item = state.inventory[target]
        refund = item.price if item.refundable else item.price * (
            1.0 - self.config.cancellation_fee_rate
        )
        state.spent = max(0.0, state.spent - refund)
        state.booked.remove(target)
        state.awaiting_revision = False
        state.client_accepted = False
        return True, False, False, "booking_removed"

    def _swap(
        self, source: int, target: int, *, require_disrupted: bool
    ) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        candidate = state.inventory[target]
        if not state.visible[target]:
            return False, False, False, "inventory_not_visible"
        if not candidate.available:
            return False, False, False, "inventory_unavailable"
        if source not in state.booked:
            return False, False, False, "source_booking_not_found"
        existing = state.inventory[source]
        if existing.category != candidate.category:
            return False, False, False, "replacement_category_mismatch"
        if require_disrupted and source not in state.active_disruption_targets():
            return False, False, False, "source_booking_not_disrupted"
        if source == target:
            return False, False, False, "same_inventory_item"
        if (
            candidate.category == InventoryCategory.ACTIVITY
            and state.activity_conflicts(candidate, ignore=existing.index)
        ):
            return False, False, False, "schedule_conflict"

        refund = existing.price if existing.refundable else existing.price * (
            1.0 - self.config.cancellation_fee_rate
        )
        projected = state.spent - refund + candidate.price
        if projected > state.hard_budget():
            return False, False, False, "hard_budget_exceeded"

        state.booked.remove(existing.index)
        state.booked.add(target)
        state.spent = projected
        state.rebooking_count += 1
        state.awaiting_revision = False
        state.client_accepted = False
        for disruption in state.disruptions:
            remaining = (
                {disruption.target_index} | disruption.affected_indices
            ) & state.booked
            if not remaining:
                disruption.resolved = True
        return True, True, False, "booking_swapped"

    def _propose(self) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        if not state.has_complete_itinerary(self.config.min_activities):
            return False, False, False, "incomplete_itinerary"
        if state.patience_remaining <= 0:
            return False, False, False, "client_patience_exhausted"
        state.proposal_count += 1
        state.patience_remaining -= 1
        utility = self.reward_model.latent_utility(state)
        state.client_accepted = utility >= self.config.acceptance_threshold
        if not state.client_accepted:
            state.awaiting_revision = True
            self._reveal_feedback()
            return True, False, False, "client_requested_changes"
        state.awaiting_revision = False
        return True, False, False, "client_accepted"

    def _message(self) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        if state.patience_remaining <= 0:
            return False, False, False, "client_patience_exhausted"
        state.patience_remaining -= 1
        changed = self._reveal_feedback()
        return (
            True,
            False,
            False,
            "preference_clarified" if changed else "no_new_information",
        )

    def _finish(self) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        if not state.has_complete_itinerary(self.config.min_activities):
            return False, False, True, "cannot_finish_incomplete_itinerary"
        if not state.client_accepted:
            return False, False, True, "client_has_not_accepted"
        state.done = True
        return True, False, True, "episode_completed"

    def _reveal_feedback(self) -> bool:
        state = self._require_state()
        hidden = np.where(
            state.feedback_revealed > 0,
            -1.0,
            state.persona.preference_weights,
        )
        index = int(np.argmax(hidden))
        if hidden[index] < 0:
            return False
        state.feedback_revealed[index] = state.persona.preference_weights[index]
        return True

    def _observation(self) -> Dict[str, np.ndarray]:
        state = self._require_state()
        request = np.array(
            [
                state.request.destination / 7.0,
                state.request.party_size / 6.0,
                state.request.duration_days / 14.0,
                min(state.request.budget / 10000.0, 1.0),
                min(state.hard_budget() / 12000.0, 1.0),
            ],
            dtype=np.float32,
        )
        preferences = np.maximum(
            state.request.stated_preferences,
            state.feedback_revealed,
        ).astype(np.float32)
        counts = state.booking_counts()
        trip_state = np.array(
            [
                min(state.spent / max(state.hard_budget(), 1.0), 1.0),
                min(
                    max(state.hard_budget() - state.spent, 0.0)
                    / max(state.hard_budget(), 1.0),
                    1.0,
                ),
                float(counts[InventoryCategory.FLIGHT] > 0),
                float(counts[InventoryCategory.HOTEL] > 0),
                min(
                    counts[InventoryCategory.ACTIVITY]
                    / max(self.config.min_activities, 1),
                    1.0,
                ),
                state.patience_remaining / max(self.config.client_patience, 1),
                min(state.step_count / self.config.max_steps, 1.0),
                float(state.client_accepted),
                min(len(state.disruptions) / 4.0, 1.0),
                float(state.awaiting_revision),
            ],
            dtype=np.float32,
        )
        inventory = np.zeros(
            (self.config.max_inventory, 15), dtype=np.float32
        )
        for item in state.inventory:
            if state.visible[item.index]:
                inventory[item.index] = self._item_features(item, state)

        booking_mask = np.zeros(self.config.max_inventory, dtype=np.int8)
        booking_mask[list(state.booked)] = 1
        disruption_mask = np.zeros(self.config.max_inventory, dtype=np.int8)
        targets = state.active_disruption_targets()
        if targets:
            disruption_mask[list(targets)] = 1

        return {
            "request": request,
            "preferences": preferences,
            "trip_state": trip_state,
            "inventory": inventory,
            "inventory_mask": state.visible.astype(np.int8),
            "booking_mask": booking_mask,
            "action_mask": self._action_mask(),
            "target_mask": self._target_mask(),
            "disruption_mask": disruption_mask,
        }

    def _item_features(
        self, item: InventoryItem, state: TravelState
    ) -> np.ndarray:
        category = np.zeros(3, dtype=np.float32)
        category[int(item.category)] = 1.0
        return np.concatenate(
            [
                category,
                np.array(
                    [
                        min(item.price / max(state.hard_budget(), 1.0), 1.0),
                        item.quality,
                        item.location,
                        item.convenience,
                    ],
                    dtype=np.float32,
                ),
                item.theme_vector.astype(np.float32),
                np.array(
                    [
                        min(
                            item.start_hour
                            / max(state.request.duration_days * 24.0, 1.0),
                            1.0,
                        ),
                        min(
                            item.end_hour
                            / max(state.request.duration_days * 24.0, 1.0),
                            1.0,
                        ),
                        float(item.refundable),
                        float(item.available),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _action_mask(self) -> np.ndarray:
        state = self._require_state()
        mask = np.zeros(len(ActionType), dtype=np.int8)
        mask[
            [
                ActionType.SEARCH_FLIGHTS,
                ActionType.SEARCH_HOTELS,
                ActionType.SEARCH_ACTIVITIES,
            ]
        ] = 1
        visible_available = [
            item
            for item in state.inventory
            if state.visible[item.index] and item.available
        ]
        for item in visible_available:
            if item.category == InventoryCategory.FLIGHT:
                mask[ActionType.SELECT_FLIGHT] = 1
            elif item.category == InventoryCategory.HOTEL:
                mask[ActionType.SELECT_HOTEL] = 1
            else:
                mask[ActionType.SELECT_ACTIVITY] = 1
        if state.booked:
            mask[ActionType.REMOVE_BOOKING] = 1
            mask[ActionType.SWAP_BOOKING] = 1
        if state.active_disruption_targets() & state.booked:
            mask[ActionType.REBOOK] = 1
        if (
            state.has_complete_itinerary(self.config.min_activities)
            and state.patience_remaining > 0
        ):
            mask[ActionType.PROPOSE_ITINERARY] = 1
        if state.patience_remaining > 0:
            mask[ActionType.MESSAGE_CLIENT] = 1
        if (
            state.has_complete_itinerary(self.config.min_activities)
            and state.client_accepted
        ):
            mask[ActionType.FINISH] = 1
        return mask

    def _target_mask(self) -> np.ndarray:
        state = self._require_state()
        mask = np.zeros(
            (len(ActionType), self.config.max_inventory),
            dtype=np.int8,
        )
        for item in state.inventory:
            index = item.index
            if state.visible[index] and item.available:
                if item.category == InventoryCategory.FLIGHT:
                    mask[ActionType.SELECT_FLIGHT, index] = 1
                elif item.category == InventoryCategory.HOTEL:
                    mask[ActionType.SELECT_HOTEL, index] = 1
                else:
                    mask[ActionType.SELECT_ACTIVITY, index] = 1

                if any(
                    state.inventory[booked].category == item.category
                    and booked != index
                    for booked in state.booked
                ):
                    mask[ActionType.SWAP_BOOKING, index] = 1

                if any(
                    state.inventory[booked].category == item.category
                    and booked in state.active_disruption_targets()
                    for booked in state.booked
                ):
                    mask[ActionType.REBOOK, index] = 1

            if index in state.booked:
                mask[ActionType.REMOVE_BOOKING, index] = 1
        return mask

    def _info(self) -> Dict[str, Any]:
        state = self._require_state()
        info = {
            "spent": round(state.spent, 2),
            "hard_budget": round(state.hard_budget(), 2),
            "steps": state.step_count,
            "searches": state.search_count,
            "proposals": state.proposal_count,
            "rebookings": state.rebooking_count,
            "invalid_actions": state.invalid_actions,
            "complete_itinerary": state.has_complete_itinerary(
                self.config.min_activities
            ),
            "client_accepted": state.client_accepted,
            "realized_satisfaction": self.reward_model.realized_satisfaction(
                state
            ),
        }
        info.update(self.reward_model.itinerary_metrics(state))
        return info

    def _require_state(self) -> TravelState:
        if self.state is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.state
