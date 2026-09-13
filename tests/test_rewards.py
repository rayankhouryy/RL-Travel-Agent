import numpy as np

from travel_env import TravelAgentEnv
from travel_env.actions import ActionType
from travel_env.config import EnvironmentConfig


def action(action_type: ActionType, target: int = 0, source: int = 0):
    return {
        "action_type": int(action_type),
        "target_index": target,
        "source_index": source,
    }


def test_repeated_search_does_not_create_positive_shaping_reward():
    env = TravelAgentEnv()
    env.reset(seed=17)

    _, first_reward, _, _, _ = env.step(action(ActionType.SEARCH_FLIGHTS))
    _, second_reward, _, _, _ = env.step(action(ActionType.SEARCH_FLIGHTS))

    assert first_reward < 0
    assert second_reward < 0


def test_booking_churn_has_negative_discounted_return():
    config = EnvironmentConfig(
        price_drift_per_step=0.0,
        depletion_probability=0.0,
        disruption_probability=0.0,
    )
    env = TravelAgentEnv(config)
    observation, _ = env.reset(seed=18)
    observation, _, _, _, _ = env.step(action(ActionType.SEARCH_FLIGHTS))
    target = int(
        np.flatnonzero(
            observation["target_mask"][ActionType.SELECT_FLIGHT]
        )[0]
    )

    _, booking_reward, _, _, _ = env.step(
        action(ActionType.SELECT_FLIGHT, target=target)
    )
    _, removal_reward, _, _, _ = env.step(
        action(ActionType.REMOVE_BOOKING, target=target)
    )

    discounted_return = (
        booking_reward + config.discount * removal_reward
    )
    assert discounted_return < 0.0


def test_budget_fit_does_not_penalize_underspending():
    env = TravelAgentEnv()
    env.reset(seed=19)
    target = (
        env.state.request.budget
        * env.state.persona.expected_budget_usage
    )

    env.state.spent = 0.7 * target
    lower_spend_score = env.reward_model.itinerary_metrics(
        env.state
    )["budget_score"]
    env.state.spent = target
    target_spend_score = env.reward_model.itinerary_metrics(
        env.state
    )["budget_score"]

    assert lower_spend_score == target_spend_score == 1.0
