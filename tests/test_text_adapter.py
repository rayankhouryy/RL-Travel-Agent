import pytest

from travel_env import TravelAgentEnv
from travel_env.actions import ActionType


def test_text_adapter_parses_parameterized_actions():
    assert TravelAgentEnv.action_from_text(
        "SWAP_BOOKING source=12 target=18"
    ) == {
        "action_type": int(ActionType.SWAP_BOOKING),
        "source_index": 12,
        "target_index": 18,
    }
    assert TravelAgentEnv.action_from_text("SEARCH_HOTELS") == {
        "action_type": int(ActionType.SEARCH_HOTELS),
        "source_index": 0,
        "target_index": 0,
    }


def test_text_adapter_rejects_missing_or_unknown_parameters():
    with pytest.raises(ValueError, match="requires target"):
        TravelAgentEnv.action_from_text("SELECT_HOTEL")
    with pytest.raises(ValueError, match="Unknown action"):
        TravelAgentEnv.action_from_text("BUY_YACHT target=3")


def test_observation_text_contains_only_visible_inventory():
    env = TravelAgentEnv()
    observation, _ = env.reset(seed=501)
    before = env.observation_to_text(observation)

    observation, *_ = env.step(
        TravelAgentEnv.action_from_text("SEARCH_HOTELS")
    )
    after = env.observation_to_text(observation)

    assert "Visible inventory:\nLegal actions:" in before
    assert "hotel" in after
    assert "flight" not in after
