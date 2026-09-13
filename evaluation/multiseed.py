from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np

from evaluation.reward_hacking import REWARD_COMPONENTS, run_episode
from policies import FlexibilityBuyerPolicy, NonRefundablePolicy
from travel_env.config import EnvironmentConfig


METRICS = (
    "reward",
    "realized",
    "sunk_fraction",
    "success",
    "spend",
    *(f"component_{name}" for name in REWARD_COMPONENTS),
)


def evaluate_blocks(
    config: EnvironmentConfig,
    blocks: int,
    episodes_per_block: int,
    seed: int,
    difficulty: int,
) -> Dict[str, Dict[str, float]]:
    block_means = defaultdict(list)
    for block in range(blocks):
        differences = defaultdict(list)
        block_seed = seed + block * 100000
        for episode in range(episodes_per_block):
            episode_seed = block_seed + episode
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
            for metric in METRICS:
                differences[metric].append(
                    flexible[metric] - nonrefundable[metric]
                )
        for metric in METRICS:
            block_means[metric].append(float(np.mean(differences[metric])))

    return {
        metric: {
            "mean": float(np.mean(values)),
            "between_block_std": float(np.std(values, ddof=1)),
            "positive_block_fraction": float(
                np.mean(np.asarray(values) > 0.0)
            ),
        }
        for metric, values in block_means.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--episodes-per-block", type=int, default=50)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    results = evaluate_blocks(
        EnvironmentConfig.from_yaml(args.config),
        blocks=args.blocks,
        episodes_per_block=args.episodes_per_block,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    print(
        f"{'metric':<18} {'mean delta':>14} "
        f"{'block std':>14} {'positive blocks':>18}"
    )
    for metric, values in results.items():
        print(
            f"{metric:<18} {values['mean']:>14.4f} "
            f"{values['between_block_std']:>14.4f} "
            f"{values['positive_block_fraction']:>18.2f}"
        )


if __name__ == "__main__":
    main()
