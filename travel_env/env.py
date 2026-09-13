from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from travel_env.actions import ActionType, TARGETED_ACTIONS, decode_action
from travel_env.config import EnvironmentConfig
from travel_env.disruptions import DisruptionEngine
from travel_env.generator import TravelWorldGenerator
from travel_env.market import MarketEngine
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
        self.market_engine = MarketEngine(self.config)
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
                "trip_state": spaces.Box(0.0, 1.0, shape=(14,), dtype=np.float32),
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
                "swap_mask": spaces.MultiBinary(
                    (2, self.config.max_inventory, self.config.max_inventory)
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
        valid, _, finishing, reason = self._apply_action(action)

        if not valid:
            state.invalid_actions += 1

        disruption = None
        try:
            decoded_action = ActionType(int(action["action_type"]))
        except (KeyError, TypeError, ValueError):
            decoded_action = None
        if valid and decoded_action == ActionType.ADVANCE_TRIP:
            disruption = self.disruption_engine.maybe_trigger(
                state, self.np_random
            )
            if disruption is not None:
                state.client_accepted = False
                state.awaiting_revision = True
                state.trip_completed = False
                state.patience_remaining = max(
                    state.patience_remaining,
                    min(2, self.config.client_patience),
                )
                reason = f"trip_disrupted:{disruption.kind}"
            elif state.current_day >= state.request.duration_days:
                state.trip_completed = True
                reason = "trip_completed"

        if valid and not state.done and decoded_action != ActionType.FINISH:
            self.market_engine.advance(state, self.np_random)

        reward = self.reward_model.transition(
            previous_potential,
            state,
            valid=valid,
            finishing=finishing,
            terminal=state.done,
        )
        terminated = state.done
        truncated = state.step_count >= state.step_limit and not terminated
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
        if action == ActionType.ADVANCE_TRIP:
            return self._advance_trip()
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
        state.booking_prices[target] = item.price
        state.spent += item.price
        state.awaiting_revision = False
        state.client_accepted = False
        state.trip_completed = False
        return True, True, False, "booking_created"

    def _remove(self, target: int) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        if target not in state.booked:
            return False, False, False, "booking_not_found"
        item = state.inventory[target]
        booked_price = state.booking_prices.pop(target, item.price)
        refund = booked_price if item.refundable else booked_price * (
            1.0 - self.config.cancellation_fee_rate
        )
        state.sunk_cost += booked_price - refund
        state.spent = max(0.0, state.spent - refund)
        state.booked.remove(target)
        state.awaiting_revision = False
        state.client_accepted = False
        state.trip_completed = False
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

        booked_price = state.booking_prices.get(source, existing.price)
        refund = booked_price if existing.refundable else booked_price * (
            1.0 - self.config.cancellation_fee_rate
        )
        projected = state.spent - refund + candidate.price
        if projected > state.hard_budget():
            return False, False, False, "hard_budget_exceeded"

        state.booked.remove(existing.index)
        state.booked.add(target)
        state.booking_prices.pop(existing.index, None)
        state.booking_prices[target] = candidate.price
        state.sunk_cost += booked_price - refund
        state.spent = projected
        state.rebooking_count += 1
        state.awaiting_revision = False
        state.client_accepted = False
        state.trip_completed = False
        for disruption in state.disruptions:
            remaining = (
                {disruption.target_index} | disruption.affected_indices
            ) & state.booked
            if not remaining:
                disruption.resolved = True
        return True, True, False, "booking_swapped"

    def _propose(self) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        if not state.has_complete_itinerary():
            return False, False, False, "incomplete_itinerary"
        if state.patience_remaining <= 0:
            return False, False, False, "client_patience_exhausted"
        state.proposal_count += 1
        state.patience_remaining -= 1
        utility = self.reward_model.latent_utility(state)
        state.client_accepted = utility >= self.config.acceptance_threshold
        if not state.client_accepted:
            self._reveal_feedback()
            if state.patience_remaining <= 0:
                state.awaiting_revision = False
                state.done = True
                return True, False, True, "client_rejected_final_proposal"
            state.awaiting_revision = True
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
        if not state.has_complete_itinerary():
            return False, False, True, "cannot_finish_incomplete_itinerary"
        if not state.client_accepted:
            return False, False, True, "client_has_not_accepted"
        if not state.trip_completed:
            return False, False, True, "trip_not_completed"
        state.done = True
        return True, False, True, "episode_completed"

    def _advance_trip(self) -> Tuple[bool, bool, bool, str]:
        state = self._require_state()
        if not state.client_accepted:
            return False, False, False, "client_has_not_accepted"
        if not state.has_complete_itinerary():
            return False, False, False, "itinerary_not_travel_ready"
        if state.trip_completed:
            return False, False, False, "trip_already_completed"
        state.trip_started = True
        state.current_day = min(
            state.current_day + 1,
            state.request.duration_days,
        )
        return True, False, False, "trip_advanced"

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
                    / max(state.required_activities, 1),
                    1.0,
                ),
                state.patience_remaining / max(self.config.client_patience, 1),
                min(state.step_count / state.step_limit, 1.0),
                float(state.client_accepted),
                min(len(state.disruptions) / 4.0, 1.0),
                float(state.awaiting_revision),
                state.current_day / max(state.request.duration_days, 1),
                float(state.trip_started),
                float(state.trip_completed),
                min(state.required_activities / 4.0, 1.0),
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
        swap_mask = self._swap_mask()
        target_mask = self._target_mask(swap_mask)
        action_mask = self._action_mask(target_mask)

        return {
            "request": request,
            "preferences": preferences,
            "trip_state": trip_state,
            "inventory": inventory,
            "inventory_mask": state.visible.astype(np.int8),
            "booking_mask": booking_mask,
            "action_mask": action_mask,
            "target_mask": target_mask,
            "swap_mask": swap_mask,
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

    def _action_mask(
        self, target_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        state = self._require_state()
        mask = np.zeros(len(ActionType), dtype=np.int8)
        if target_mask is None:
            target_mask = self._target_mask()
        mask[
            [
                ActionType.SEARCH_FLIGHTS,
                ActionType.SEARCH_HOTELS,
                ActionType.SEARCH_ACTIVITIES,
            ]
        ] = 1
        for action in (
            ActionType.SELECT_FLIGHT,
            ActionType.SELECT_HOTEL,
            ActionType.SELECT_ACTIVITY,
            ActionType.REMOVE_BOOKING,
            ActionType.SWAP_BOOKING,
            ActionType.REBOOK,
        ):
            mask[action] = int(bool(target_mask[action].any()))
        if state.booked and target_mask[ActionType.REMOVE_BOOKING].any():
            mask[ActionType.REMOVE_BOOKING] = 1
        if (
            state.has_complete_itinerary()
            and state.patience_remaining > 0
        ):
            mask[ActionType.PROPOSE_ITINERARY] = 1
        if state.patience_remaining > 0:
            mask[ActionType.MESSAGE_CLIENT] = 1
        if (
            state.has_complete_itinerary()
            and state.client_accepted
            and not state.trip_completed
        ):
            mask[ActionType.ADVANCE_TRIP] = 1
        if (
            state.has_complete_itinerary()
            and state.client_accepted
            and state.trip_completed
        ):
            mask[ActionType.FINISH] = 1
        return mask

    def _target_mask(
        self, swap_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        state = self._require_state()
        if swap_mask is None:
            swap_mask = self._swap_mask()
        mask = np.zeros(
            (len(ActionType), self.config.max_inventory),
            dtype=np.int8,
        )
        for item in state.inventory:
            index = item.index
            if state.visible[index] and item.available:
                if self._is_selectable(item, InventoryCategory.FLIGHT):
                    mask[ActionType.SELECT_FLIGHT, index] = 1
                if self._is_selectable(item, InventoryCategory.HOTEL):
                    mask[ActionType.SELECT_HOTEL, index] = 1
                if self._is_selectable(item, InventoryCategory.ACTIVITY):
                    mask[ActionType.SELECT_ACTIVITY, index] = 1

                if swap_mask[0, :, index].any():
                    mask[ActionType.SWAP_BOOKING, index] = 1

                if swap_mask[1, :, index].any():
                    mask[ActionType.REBOOK, index] = 1

            if index in state.booked:
                mask[ActionType.REMOVE_BOOKING, index] = 1
        return mask

    def _is_selectable(
        self, item: InventoryItem, category: InventoryCategory
    ) -> bool:
        state = self._require_state()
        if (
            item.category != category
            or item.index in state.booked
            or not item.available
        ):
            return False
        if (
            category in (InventoryCategory.FLIGHT, InventoryCategory.HOTEL)
            and state.booking_for(category) is not None
        ):
            return False
        if (
            category == InventoryCategory.ACTIVITY
            and state.activity_conflicts(item)
        ):
            return False
        return state.spent + item.price <= state.hard_budget()

    def _swap_mask(self) -> np.ndarray:
        state = self._require_state()
        mask = np.zeros(
            (2, self.config.max_inventory, self.config.max_inventory),
            dtype=np.int8,
        )
        disrupted = state.active_disruption_targets()
        for source in state.booked:
            for candidate in state.inventory:
                if self._is_valid_swap_pair(source, candidate):
                    mask[0, source, candidate.index] = 1
                    if source in disrupted:
                        mask[1, source, candidate.index] = 1
        return mask

    def _is_valid_swap_pair(
        self, source: int, candidate: InventoryItem
    ) -> bool:
        state = self._require_state()
        existing = state.inventory[source]
        if (
            not state.visible[candidate.index]
            or not candidate.available
            or existing.category != candidate.category
            or source == candidate.index
        ):
            return False
        if (
            candidate.category == InventoryCategory.ACTIVITY
            and state.activity_conflicts(candidate, ignore=source)
        ):
            return False
        refund = (
            state.booking_prices.get(source, existing.price)
            if existing.refundable
            else state.booking_prices.get(source, existing.price)
            * (1.0 - self.config.cancellation_fee_rate)
        )
        return state.spent - refund + candidate.price <= state.hard_budget()

    def _info(self) -> Dict[str, Any]:
        state = self._require_state()
        info = {
            "spent": round(state.spent, 2),
            "sunk_cost": round(state.sunk_cost, 2),
            "sunk_cost_fraction": state.sunk_cost
            / max(state.hard_budget(), 1.0),
            "hard_budget": round(state.hard_budget(), 2),
            "steps": state.step_count,
            "step_limit": state.step_limit,
            "searches": state.search_count,
            "proposals": state.proposal_count,
            "rebookings": state.rebooking_count,
            "invalid_actions": state.invalid_actions,
            "complete_itinerary": state.has_complete_itinerary(),
            "client_accepted": state.client_accepted,
            "difficulty": state.difficulty,
            "required_activities": state.required_activities,
            "current_day": state.current_day,
            "trip_started": state.trip_started,
            "trip_completed": state.trip_completed,
            "disruptions": len(state.disruptions),
            "market_steps": state.market_steps,
            "depleted_inventory": state.depleted_inventory,
            "refundable_share": (
                float(
                    np.mean(
                        [
                            float(item.refundable)
                            for item in state.booked_items()
                        ]
                    )
                )
                if state.booked
                else 0.0
            ),
            "realized_satisfaction": self.reward_model.realized_satisfaction(
                state
            ),
        }
        info.update(self.reward_model.itinerary_metrics(state))
        return info

    def observation_to_text(
        self, observation: Optional[Dict[str, np.ndarray]] = None
    ) -> str:
        state = self._require_state()
        observation = observation or self._observation()
        visible_preferences = [
            f"{theme}={observation['preferences'][index]:.2f}"
            for index, theme in enumerate(THEMES)
            if observation["preferences"][index] > 0
        ]
        lines = [
            (
                f"Trip to destination {state.request.destination} for "
                f"{state.request.party_size} travelers and "
                f"{state.request.duration_days} days."
            ),
            (
                f"Budget ${state.request.budget:,.2f}; "
                f"spent ${state.spent:,.2f}; "
                f"hard limit ${state.hard_budget():,.2f}."
            ),
            (
                f"Difficulty {state.difficulty}; day {state.current_day}/"
                f"{state.request.duration_days}; patience "
                f"{state.patience_remaining}."
            ),
            (
                "Known preferences: "
                + (", ".join(visible_preferences) or "none")
            ),
            "Visible inventory:",
        ]
        for item in state.inventory:
            if not state.visible[item.index]:
                continue
            status = "available" if item.available else "unavailable"
            booked = " booked" if item.index in state.booked else ""
            lines.append(
                f"  [{item.index}] {item.category.name.lower()} "
                f"${item.price:,.2f} quality={item.quality:.2f} "
                f"location={item.location:.2f} "
                f"convenience={item.convenience:.2f} "
                f"{'refundable' if item.refundable else 'nonrefundable'} "
                f"{status}{booked}"
            )
        legal = [
            action.name
            for action in ActionType
            if observation["action_mask"][action]
        ]
        lines.append("Legal actions: " + ", ".join(legal))
        return "\n".join(lines)

    @staticmethod
    def action_from_text(text: str) -> Dict[str, int]:
        normalized = text.strip().upper()
        if not normalized:
            raise ValueError("Action text cannot be empty")

        action_name = normalized.split()[0]
        try:
            action = ActionType[action_name]
        except KeyError as exc:
            raise ValueError(f"Unknown action type: {action_name}") from exc

        def parameter(name: str, default: Optional[int] = None) -> int:
            match = re.search(rf"\b{name}\s*=\s*(\d+)\b", normalized)
            if match:
                return int(match.group(1))
            if default is not None:
                return default
            raise ValueError(f"{action.name} requires {name.lower()}=<index>")

        source = 0
        target = 0
        if action in {
            ActionType.SELECT_FLIGHT,
            ActionType.SELECT_HOTEL,
            ActionType.SELECT_ACTIVITY,
            ActionType.REMOVE_BOOKING,
        }:
            target = parameter("TARGET")
        elif action in {ActionType.SWAP_BOOKING, ActionType.REBOOK}:
            source = parameter("SOURCE")
            target = parameter("TARGET")

        return {
            "action_type": int(action),
            "source_index": source,
            "target_index": target,
        }

    def _require_state(self) -> TravelState:
        if self.state is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self.state
