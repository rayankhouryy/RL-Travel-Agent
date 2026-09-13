from dataclasses import replace

import pytest

from policies import HeuristicPolicy
from travel_env import TravelAgentEnv
from travel_env.actions import ActionType
from travel_env.config import EnvironmentConfig


def discounted_trace(config, seed):
    env = TravelAgentEnv(config)
    policy = HeuristicPolicy()
    observation, _ = env.reset(seed=seed, options={"difficulty": 1})
    rewards = []
    terminated = truncated = False
    while not (terminated or truncated):
        observation, reward, terminated, truncated, _ = env.step(
            policy.act(observation)
        )
        rewards.append(reward)
    assert terminated
    assert not truncated
    return sum(
        (config.discount**step) * reward
        for step, reward in enumerate(rewards)
    )


def test_potential_is_zero_at_reset():
    env = TravelAgentEnv()
    env.reset(seed=300)

    assert env.reward_model.potential(env.state) == 0.0


def test_potential_shaping_preserves_discounted_return():
    base = EnvironmentConfig(disruption_probability=0.0)
    returns = []
    for scale in (0.0, 0.25, 1.0):
        reward = replace(base.reward, shaping_scale=scale)
        config = replace(base, reward=reward)
        returns.append(discounted_trace(config, seed=301))

    assert max(returns) - min(returns) < 1e-9


def test_terminal_transition_applies_final_shaping_term():
    config = EnvironmentConfig(disruption_probability=0.0)
    env = TravelAgentEnv(config)
    policy = HeuristicPolicy()
    observation, _ = env.reset(seed=304, options={"difficulty": 1})

    while True:
        selected = policy.act(observation)
        if selected["action_type"] == int(ActionType.FINISH):
            previous_potential = env.reward_model.potential(env.state)
            _, reward, terminated, truncated, info = env.step(selected)
            break
        observation, _, terminated, truncated, _ = env.step(selected)
        assert not terminated
        assert not truncated

    assert terminated
    assert not truncated
    assert info["reward_components"]["potential_shaping"] == pytest.approx(
        -config.reward.shaping_scale * previous_potential
    )
    assert reward == pytest.approx(
        sum(info["reward_components"].values())
    )


def test_realized_satisfaction_is_held_out(monkeypatch):
    config = EnvironmentConfig(disruption_probability=0.0)

    def reward_trace(stub_realized):
        env = TravelAgentEnv(config)
        if stub_realized:
            monkeypatch.setattr(
                env.reward_model,
                "realized_satisfaction",
                lambda state: 0.0,
            )
        policy = HeuristicPolicy()
        observation, _ = env.reset(seed=302, options={"difficulty": 1})
        rewards = []
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, _ = env.step(
                policy.act(observation)
            )
            rewards.append(reward)
        return rewards

    assert reward_trace(False) == pytest.approx(reward_trace(True), abs=0.0)


def test_realized_satisfaction_is_independent_of_latent_utility(monkeypatch):
    env = TravelAgentEnv(EnvironmentConfig(disruption_probability=0.0))
    policy = HeuristicPolicy()
    observation, _ = env.reset(seed=303, options={"difficulty": 1})
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(
            policy.act(observation)
        )

    baseline = env.reward_model.realized_satisfaction(env.state)
    monkeypatch.setattr(
        env.reward_model,
        "latent_utility",
        lambda state: 0.0,
    )

    assert env.reward_model.realized_satisfaction(env.state) == baseline


def test_disruption_sensitivity_increases_value_of_robustness():
    env = TravelAgentEnv(EnvironmentConfig(disruption_probability=0.0))
    policy = HeuristicPolicy()
    observation, _ = env.reset(seed=305, options={"difficulty": 1})
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(
            policy.act(observation)
        )

    env.state.persona.disruption_sensitivity = 0.0
    tolerant_score = env.reward_model.realized_satisfaction(env.state)
    env.state.persona.disruption_sensitivity = 1.0
    sensitive_score = env.reward_model.realized_satisfaction(env.state)

    assert sensitive_score > tolerant_score


def test_successful_terminal_reward_excludes_structural_invariants():
    env = TravelAgentEnv(EnvironmentConfig(disruption_probability=0.0))
    policy = HeuristicPolicy()
    observation, _ = env.reset(seed=306, options={"difficulty": 1})
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(
            policy.act(observation)
        )

    components = env.reward_model.terminal_components(env.state)
    assert "coherence" not in components
    assert "violations" not in components


def test_terminal_preference_reward_depends_on_hidden_persona():
    env = TravelAgentEnv(EnvironmentConfig(disruption_probability=0.0))
    policy = HeuristicPolicy()
    observation, _ = env.reset(seed=307, options={"difficulty": 1})
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(
            policy.act(observation)
        )

    env.state.persona.quality_preference = 0.1
    low_quality_priority = env.reward_model.terminal_components(
        env.state
    )["preference"]
    env.state.persona.quality_preference = 3.0
    high_quality_priority = env.reward_model.terminal_components(
        env.state
    )["preference"]

    assert high_quality_priority != low_quality_priority
