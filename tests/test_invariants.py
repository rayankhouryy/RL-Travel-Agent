import numpy as np
import pytest

from travel_env import TravelAgentEnv
from travel_env.actions import ActionType
from travel_env.config import EnvironmentConfig
from travel_env.curriculum import difficulty_profile
from travel_env.models import InventoryCategory


def action(action_type: ActionType, target: int = 0, source: int = 0):
    return {
        "action_type": int(action_type),
        "target_index": target,
        "source_index": source,
    }


def reveal_all(env):
    for action_type in (
        ActionType.SEARCH_FLIGHTS,
        ActionType.SEARCH_HOTELS,
        ActionType.SEARCH_ACTIVITIES,
    ):
        env.step(action(action_type))


def complete_state_directly(env):
    state = env.state
    flight = next(
        item
        for item in state.inventory
        if item.category == InventoryCategory.FLIGHT and item.available
    )
    hotel = next(
        item
        for item in state.inventory
        if item.category == InventoryCategory.HOTEL and item.available
    )
    activities = []
    for item in state.inventory:
        if item.category != InventoryCategory.ACTIVITY or not item.available:
            continue
        if all(
            item.end_hour <= chosen.start_hour
            or chosen.end_hour <= item.start_hour
            for chosen in activities
        ):
            activities.append(item)
        if len(activities) == state.required_activities:
            break
    chosen = [flight, hotel, *activities]
    state.booked = {item.index for item in chosen}
    state.spent = sum(item.price for item in chosen)
    return chosen


def test_curriculum_changes_task_requirements_and_observability():
    env = TravelAgentEnv(EnvironmentConfig(disruption_probability=0.08))

    easy, _ = env.reset(seed=12, options={"difficulty": 1})
    easy_state = env.state
    hard, _ = env.reset(seed=12, options={"difficulty": 4})
    hard_state = env.state

    assert easy_state.required_activities == 1
    assert hard_state.required_activities == 4
    assert np.count_nonzero(easy["preferences"]) == 3
    assert np.count_nonzero(hard["preferences"]) == 1
    assert easy_state.disruption_probability == 0.0
    assert hard_state.disruption_probability > 0.0
    assert hard_state.step_limit > easy_state.step_limit


def test_invalid_difficulty_is_rejected():
    env = TravelAgentEnv()

    with pytest.raises(ValueError, match="difficulty"):
        env.reset(seed=1, options={"difficulty": 9})


def test_finish_requires_trip_completion_after_acceptance():
    env = TravelAgentEnv()
    env.reset(seed=20)
    complete_state_directly(env)
    env.state.client_accepted = True

    _, reward, terminated, _, info = env.step(action(ActionType.FINISH))

    assert not terminated
    assert not info["action_valid"]
    assert info["action_result"] == "trip_not_completed"
    assert reward < 0


def test_advance_trip_requires_client_acceptance():
    env = TravelAgentEnv()
    env.reset(seed=21)
    complete_state_directly(env)

    _, _, terminated, _, info = env.step(action(ActionType.ADVANCE_TRIP))

    assert not terminated
    assert not info["action_valid"]
    assert info["action_result"] == "client_has_not_accepted"
    assert env.state.current_day == 0


def test_final_client_rejection_terminates_as_failure():
    env = TravelAgentEnv(
        EnvironmentConfig(
            acceptance_threshold=1.0,
            client_patience=1,
        )
    )
    env.reset(seed=22)
    complete_state_directly(env)
    env.state.patience_remaining = 1

    _, reward, terminated, _, info = env.step(
        action(ActionType.PROPOSE_ITINERARY)
    )

    assert terminated
    assert not info["client_accepted"]
    assert info["action_result"] == "client_rejected_final_proposal"
    assert reward < 0


def test_non_refundable_cancellation_charges_fee():
    config = EnvironmentConfig(cancellation_fee_rate=0.20)
    env = TravelAgentEnv(config)
    env.reset(seed=23)
    state = env.state
    item = next(
        item
        for item in state.inventory
        if item.available and not item.refundable
    )
    state.booked.add(item.index)
    state.spent = item.price

    _, _, _, _, info = env.step(
        action(ActionType.REMOVE_BOOKING, target=item.index)
    )

    assert info["action_valid"]
    assert state.spent == pytest.approx(item.price * 0.20)


def test_overlapping_activity_is_rejected_and_masked():
    env = TravelAgentEnv()
    observation, _ = env.reset(seed=24)
    observation, *_ = env.step(action(ActionType.SEARCH_ACTIVITIES))
    state = env.state
    first = next(
        item
        for item in state.inventory
        if item.category == InventoryCategory.ACTIVITY and item.available
    )
    overlapping = next(
        item
        for item in state.inventory
        if item.category == InventoryCategory.ACTIVITY
        and item.available
        and item.index != first.index
    )
    overlapping.start_hour = first.start_hour + 0.5
    overlapping.end_hour = first.end_hour - 0.5

    env.step(action(ActionType.SELECT_ACTIVITY, target=first.index))
    observation = env._observation()

    assert not observation["target_mask"][
        ActionType.SELECT_ACTIVITY, overlapping.index
    ]
    _, _, _, _, info = env.step(
        action(ActionType.SELECT_ACTIVITY, target=overlapping.index)
    )
    assert not info["action_valid"]
    assert info["action_result"] == "schedule_conflict"


def test_unaffordable_inventory_is_not_a_valid_target():
    env = TravelAgentEnv()
    env.reset(seed=25)
    reveal_all(env)
    state = env.state
    state.spent = state.hard_budget() - 1.0

    observation = env._observation()

    assert not observation["target_mask"][
        ActionType.SELECT_FLIGHT
    ].any()
    assert not observation["target_mask"][
        ActionType.SELECT_HOTEL
    ].any()
    assert not observation["target_mask"][
        ActionType.SELECT_ACTIVITY
    ].any()


def test_difficulty_profiles_are_complete_and_ordered():
    profiles = [difficulty_profile(level) for level in range(1, 5)]

    assert [profile.level for profile in profiles] == [1, 2, 3, 4]
    assert [profile.required_activities for profile in profiles] == [1, 2, 3, 4]
    assert [profile.max_disruptions for profile in profiles] == [0, 1, 2, 3]
    assert [profile.step_limit for profile in profiles] == sorted(
        profile.step_limit for profile in profiles
    )

