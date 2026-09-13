import numpy as np

from travel_env import TravelAgentEnv
from travel_env.actions import ActionType


def action(action_type: ActionType, target: int = 0, source: int = 0):
    return {
        "action_type": int(action_type),
        "target_index": target,
        "source_index": source,
    }


def test_reset_is_deterministic_and_observation_is_valid():
    env = TravelAgentEnv()

    first, first_info = env.reset(seed=123)
    second, second_info = env.reset(seed=123)

    assert env.observation_space.contains(first)
    assert env.observation_space.contains(second)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first_info == second_info


def test_inventory_requires_search_before_booking():
    env = TravelAgentEnv()
    env.reset(seed=4)

    _, reward, terminated, truncated, info = env.step(
        action(ActionType.SELECT_FLIGHT, 0)
    )

    assert not info["action_valid"]
    assert info["action_result"] == "inventory_not_visible"
    assert reward < 0
    assert not terminated
    assert not truncated


def test_search_reveals_only_requested_category():
    env = TravelAgentEnv()
    env.reset(seed=9)

    observation, _, _, _, info = env.step(action(ActionType.SEARCH_HOTELS))

    visible = np.flatnonzero(observation["inventory_mask"])
    assert info["action_valid"]
    assert len(visible) > 0
    assert all(env.state.inventory[index].category.value == 1 for index in visible)


def test_finish_requires_complete_accepted_itinerary():
    env = TravelAgentEnv()
    env.reset(seed=2)

    _, reward, terminated, _, info = env.step(action(ActionType.FINISH))

    assert not terminated
    assert not info["action_valid"]
    assert info["action_result"] == "cannot_finish_incomplete_itinerary"
    assert reward <= -3.0
