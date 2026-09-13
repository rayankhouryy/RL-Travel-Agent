from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from typing import Dict

from policies import HeuristicPolicy
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig, RewardWeights


PROFILES = {
    "budget": RewardWeights(
        preference=1.0,
        coherence=2.0,
        budget=5.0,
        quality=1.0,
        convenience=1.0,
    ),
    "balanced": RewardWeights(),
    "experience": RewardWeights(
        preference=5.0,
        coherence=2.0,
        budget=1.0,
        quality=3.0,
        convenience=2.0,
    ),
}


def run_profile(
    name: str,
    base_config: EnvironmentConfig,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    config = replace(base_config, reward=PROFILES[name])
    env = TravelAgentEnv(config)
    policy = HeuristicPolicy(profile=name)
    totals = defaultdict(float)

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        reward_total = 0.0
        info = {}
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(
                policy.act(observation)
            )
            reward_total += reward

        totals["reward"] += reward_total
        totals["success"] += float(terminated)
        for metric in (
            "spent",
            "quality",
            "preference_match",
            "convenience",
            "realized_satisfaction",
        ):
            totals[metric] += info[metric]

    return {key: value / episodes for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = EnvironmentConfig.from_yaml(args.config)
    results = {
        name: run_profile(name, config, args.episodes, args.seed)
        for name in PROFILES
    }
    metrics = (
        "spent",
        "quality",
        "preference_match",
        "convenience",
        "realized_satisfaction",
        "success",
        "reward",
    )
    print(f"{'metric':<24}" + "".join(f"{name:>14}" for name in results))
    for metric in metrics:
        print(
            f"{metric:<24}"
            + "".join(
                f"{results[name][metric]:>14.3f}" for name in results
            )
        )


if __name__ == "__main__":
    main()

