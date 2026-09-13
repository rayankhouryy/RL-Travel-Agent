from dataclasses import replace

import pytest

from policies import HeuristicPolicy
from travel_env import TravelAgentEnv
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
