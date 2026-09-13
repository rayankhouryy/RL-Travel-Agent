from __future__ import annotations

import argparse
from collections import defaultdict
from math import sqrt
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from policies import (
    CheapestPolicy,
    FlexibilityBuyerPolicy,
    HeuristicPolicy,
    NonRefundablePolicy,
    QualityMaximizerPolicy,
    RandomPolicy,
)
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig


POLICIES = {
    "random": lambda seed: RandomPolicy(seed=seed),
    "cheapest": lambda seed: CheapestPolicy(),
    "quality_maximizer": lambda seed: QualityMaximizerPolicy(),
    "nonrefundable": lambda seed: NonRefundablePolicy(),
    "flexibility_buyer": lambda seed: FlexibilityBuyerPolicy(),
    "heuristic": lambda seed: HeuristicPolicy(),
}


def run_episode(
    config: EnvironmentConfig,
    policy_factory: Callable[[int], object],
    seed: int,
    difficulty: int,
) -> Dict[str, float]:
    env = TravelAgentEnv(config)
    policy = policy_factory(seed)
    observation, _ = env.reset(seed=seed, options={"difficulty": difficulty})
    total_reward = 0.0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(
            policy.act(observation)
        )
        total_reward += reward
    return {
        "reward": total_reward,
        "success": float(terminated and info["client_accepted"]),
        "realized": info["realized_satisfaction"],
        "spend": info["spent"],
        "quality": info["quality"],
        "preference": info["preference_match"],
        "refundable_share": info["refundable_share"],
        "sunk_fraction": info["sunk_cost_fraction"],
    }


def aggregate(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def evaluate_policies(
    config: EnvironmentConfig,
    episodes: int,
    seed: int,
    difficulty: int,
) -> Dict[str, Dict[str, float]]:
    return {
        name: aggregate(
            run_episode(config, factory, seed + episode, difficulty)
            for episode in range(episodes)
        )
        for name, factory in POLICIES.items()
    }


def rank(values: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(values, key=values.get, reverse=True)
    return {name: index + 1 for index, name in enumerate(ordered)}


def spearman(
    first: Dict[str, float], second: Dict[str, float]
) -> float:
    first_rank = rank(first)
    second_rank = rank(second)
    count = len(first_rank)
    squared = sum(
        (first_rank[name] - second_rank[name]) ** 2
        for name in first_rank
    )
    return 1.0 - 6.0 * squared / (count * (count**2 - 1))


def confidence_interval(values: List[float]) -> Tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    half_width = 1.96 * float(np.std(values, ddof=1)) / sqrt(len(values))
    return mean, half_width


def fragility_probe(
    config: EnvironmentConfig,
    episodes: int,
    seed: int,
    difficulty: int,
) -> Dict[str, Tuple[float, float]]:
    differences = defaultdict(list)
    for episode in range(episodes):
        episode_seed = seed + episode
        flexible = run_episode(
            config,
            lambda _: FlexibilityBuyerPolicy(),
            episode_seed,
            difficulty,
        )
        nonrefundable = run_episode(
            config,
            lambda _: NonRefundablePolicy(),
            episode_seed,
            difficulty,
        )
        for metric in (
            "reward",
            "realized",
            "sunk_fraction",
            "refundable_share",
            "spend",
        ):
            differences[metric].append(
                flexible[metric] - nonrefundable[metric]
            )
    return {
        metric: confidence_interval(values)
        for metric, values in differences.items()
    }


def print_policy_table(results: Dict[str, Dict[str, float]]) -> None:
    metrics = (
        "reward",
        "success",
        "realized",
        "spend",
        "quality",
        "preference",
        "refundable_share",
        "sunk_fraction",
    )
    print(f"{'policy':<22}" + "".join(f"{metric:>18}" for metric in metrics))
    for name, row in results.items():
        print(
            f"{name:<22}"
            + "".join(f"{row[metric]:>18.3f}" for metric in metrics)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--probe-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = EnvironmentConfig.from_yaml(args.config)
    results = evaluate_policies(
        config,
        episodes=args.episodes,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    print_policy_table(results)

    reward_ranking = {
        name: row["reward"] for name, row in results.items()
    }
    realized_ranking = {
        name: row["realized"] for name, row in results.items()
    }
    print(
        "\nSpearman(reward, realized) = "
        f"{spearman(reward_ranking, realized_ranking):+.3f}"
    )

    print("\nPaired flexibility minus non-refundable:")
    probe = fragility_probe(
        config,
        episodes=args.probe_episodes,
        seed=args.seed + 10000,
        difficulty=args.difficulty,
    )
    for metric, (mean, half_width) in probe.items():
        print(f"  delta {metric:<18} {mean:+.4f} +/- {half_width:.4f}")


if __name__ == "__main__":
    main()
