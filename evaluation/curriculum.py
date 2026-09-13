from __future__ import annotations

import argparse

from evaluation.evaluate import evaluate
from travel_env.config import EnvironmentConfig
from travel_env.curriculum import DIFFICULTY_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    base = EnvironmentConfig.from_yaml(args.config)
    results = {}
    for level in DIFFICULTY_PROFILES:
        config = EnvironmentConfig(
            **{
                **base.__dict__,
                "difficulty": level,
            }
        )
        results[level] = evaluate(
            "heuristic",
            episodes=args.episodes,
            seed=args.seed,
            config=config,
        )

    metrics = (
        "success",
        "reward",
        "satisfaction",
        "steps",
        "rebookings",
        "invalid_actions",
    )
    print(f"{'metric':<24}" + "".join(f"{level:>12}" for level in results))
    for metric in metrics:
        print(
            f"{metric:<24}"
            + "".join(
                f"{results[level][metric]:>12.3f}" for level in results
            )
        )


if __name__ == "__main__":
    main()

