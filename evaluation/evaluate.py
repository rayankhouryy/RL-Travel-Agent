from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict

import numpy as np

from policies import HeuristicPolicy, RandomPolicy
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig


def evaluate(
    policy_name: str,
    episodes: int,
    seed: int,
    config: EnvironmentConfig,
) -> Dict[str, float]:
    env = TravelAgentEnv(config)
    policy = (
        HeuristicPolicy()
        if policy_name == "heuristic"
        else RandomPolicy(seed=seed)
    )
    totals = defaultdict(float)

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(
                policy.act(observation)
            )
            episode_reward += reward

        totals["episodes"] += 1
        totals["success"] += float(terminated and info["client_accepted"])
        totals["reward"] += episode_reward
        totals["satisfaction"] += info["realized_satisfaction"]
        totals["spend"] += info["spent"]
        totals["preference_match"] += info["preference_match"]
        totals["quality"] += info["quality"]
        totals["convenience"] += info["convenience"]
        totals["recovery"] += info["recovery"]
        totals["violations"] += info["violations"]
        totals["budget_compliance"] += float(
            info["spent"] <= info["hard_budget"]
        )
        totals["complete_itinerary"] += float(info["complete_itinerary"])
        totals["invalid_actions"] += info["invalid_actions"]
        totals["steps"] += info["steps"]
        totals["rebookings"] += info["rebookings"]

    count = totals.pop("episodes")
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("random", "heuristic", "both"),
        default="both",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = EnvironmentConfig.from_yaml(args.config)
    policies = (
        ("random", "heuristic")
        if args.policy == "both"
        else (args.policy,)
    )
    results = {
        policy: evaluate(policy, args.episodes, args.seed, config)
        for policy in policies
    }

    metrics = sorted(next(iter(results.values())))
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

