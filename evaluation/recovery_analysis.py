from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from evaluation.reward_hacking import run_episode
from policies import HeuristicPolicy
from travel_env.config import EnvironmentConfig


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "episodes": float(len(rows)),
        "terminal_utility_mean": float(
            np.mean([row["component_client_utility"] for row in rows])
        ),
        "terminal_utility_std": float(
            np.std(
                [row["component_client_utility"] for row in rows],
                ddof=1,
            )
        )
        if len(rows) > 1
        else 0.0,
        "realized_mean": float(np.mean([row["realized"] for row in rows])),
        "steps_mean": float(np.mean([row["steps"] for row in rows])),
        "sunk_fraction_mean": float(
            np.mean([row["sunk_fraction"] for row in rows])
        ),
    }


def evaluate_recovery(
    config: EnvironmentConfig,
    episodes: int,
    seed: int,
    difficulty: int,
) -> Dict[str, Dict[str, float]]:
    successful = []
    for episode in range(episodes):
        row = run_episode(
            config,
            lambda _: HeuristicPolicy(),
            seed + episode,
            difficulty,
        )
        if row["success"]:
            successful.append(row)

    groups = {
        "no_disruption": [
            row for row in successful if row["disruptions"] == 0
        ],
        "recovered_disruption": [
            row for row in successful if row["disruptions"] > 0
        ],
    }
    if any(not rows for rows in groups.values()):
        raise RuntimeError(
            "Recovery analysis requires successful episodes in both groups"
        )
    return {name: summarize(rows) for name, rows in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--difficulty", type=int, default=4)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    results = evaluate_recovery(
        EnvironmentConfig.from_yaml(args.config),
        episodes=args.episodes,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    metrics = next(iter(results.values())).keys()
    print(f"{'metric':<24}" + "".join(f"{name:>24}" for name in results))
    for metric in metrics:
        print(
            f"{metric:<24}"
            + "".join(
                f"{results[name][metric]:>24.4f}" for name in results
            )
        )


if __name__ == "__main__":
    main()
