import copy

import numpy as np
import pytest

from policies import RandomPolicy
from travel_env import TravelAgentEnv
from travel_env.actions import ActionType, TARGETED_ACTIONS
from travel_env.config import EnvironmentConfig


def action(action_type, target=0, source=0):
    return {
        "action_type": int(action_type),
        "target_index": int(target),
        "source_index": int(source),
    }


def assert_advertised_actions_are_valid(env, observation):
    for action_type in ActionType:
        if not observation["action_mask"][action_type]:
            continue
        if action_type in (ActionType.SWAP_BOOKING, ActionType.REBOOK):
            operation = 0 if action_type == ActionType.SWAP_BOOKING else 1
            pairs = np.argwhere(observation["swap_mask"][operation])
            for source, target in pairs[:8]:
                probe = copy.deepcopy(env)
                _, _, _, _, info = probe.step(
                    action(action_type, target=target, source=source)
                )
                assert info["action_valid"], (
                    action_type,
                    source,
                    target,
                    info["action_result"],
                )
        elif action_type in TARGETED_ACTIONS:
            targets = np.flatnonzero(
                observation["target_mask"][action_type]
            )
            for target in targets[:8]:
                probe = copy.deepcopy(env)
                _, _, _, _, info = probe.step(
                    action(action_type, target=target)
                )
                assert info["action_valid"], (
                    action_type,
                    target,
                    info["action_result"],
                )
        else:
            probe = copy.deepcopy(env)
            _, _, _, _, info = probe.step(action(action_type))
            assert info["action_valid"], (
                action_type,
                info["action_result"],
            )


def test_masks_are_sound_across_randomized_states():
    config = EnvironmentConfig(disruption_probability=0.08)
    for seed in range(20):
        env = TravelAgentEnv(config)
        policy = RandomPolicy(seed=seed)
        observation, _ = env.reset(
            seed=50000 + seed,
            options={"difficulty": 1 + seed % 4},
        )
        for _ in range(12):
            assert_advertised_actions_are_valid(env, observation)
            observation, _, terminated, truncated, _ = env.step(
                policy.act(observation)
            )
            if terminated or truncated:
                break


def test_random_valid_transitions_preserve_core_invariants():
    config = EnvironmentConfig(disruption_probability=0.08)
    for seed in range(40):
        env = TravelAgentEnv(config)
        policy = RandomPolicy(seed=seed + 900)
        observation, _ = env.reset(
            seed=60000 + seed,
            options={"difficulty": 1 + seed % 4},
        )
        previous_booking_prices = {}
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(
                policy.act(observation)
            )
            assert reward == pytest.approx(
                sum(info["reward_components"].values())
            )
            state = env.state
            assert state.spent <= state.hard_budget() + 1e-6
            active_disruptions = state.active_disruption_targets()
            assert all(
                state.inventory[index].available
                or index in active_disruptions
                for index in state.booked
            )
            for index, price in state.booking_prices.items():
                if index in previous_booking_prices:
                    assert price == previous_booking_prices[index]
            previous_booking_prices = dict(state.booking_prices)
            activities = sorted(
                (
                    state.inventory[index]
                    for index in state.booked
                    if state.inventory[index].category.value == 2
                ),
                key=lambda item: item.start_hour,
            )
            assert all(
                current.end_hour <= following.start_hour
                for current, following in zip(
                    activities,
                    activities[1:],
                )
            )
            if terminated and info["client_accepted"]:
                assert info["trip_completed"]
                assert info["complete_itinerary"]


def test_identical_seed_and_actions_produce_identical_trace():
    config = EnvironmentConfig(disruption_probability=0.08)

    def trace():
        env = TravelAgentEnv(config)
        policy = RandomPolicy(seed=777)
        observation, _ = env.reset(
            seed=70000,
            options={"difficulty": 3},
        )
        output = []
        terminated = truncated = False
        while not (terminated or truncated):
            selected = policy.act(observation)
            observation, reward, terminated, truncated, info = env.step(
                selected
            )
            output.append(
                (
                    selected,
                    reward,
                    info["spent"],
                    info["depleted_inventory"],
                    info["action_result"],
                )
            )
        return output

    assert trace() == trace()
