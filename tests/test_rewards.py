from travel_env import TravelAgentEnv
from travel_env.actions import ActionType


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
