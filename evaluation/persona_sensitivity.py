from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict

import numpy as np

from evaluation.oracle_regret import exact_slot_oracle
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig


def evaluate_persona_sensitivity(
    config: EnvironmentConfig,
    worlds: int,
    personas_per_world: int,
    seed: int,
    difficulty: int,
) -> Dict[str, float]:
    changed_worlds = 0
    unique_counts = []

    for world_index in range(worlds):
        world_seed = seed + world_index
        env = TravelAgentEnv(config)
        env.reset(seed=world_seed, options={"difficulty": difficulty})
        base_flexibility = env.state.persona.budget_flexibility
        choices = set()

        for persona_index in range(personas_per_world):
            donor = TravelAgentEnv(config)
            donor.reset(
                seed=seed + 100000 + world_index * personas_per_world
                + persona_index,
                options={"difficulty": difficulty},
            )
            env.state.persona = replace(
                donor.state.persona,
                budget_flexibility=base_flexibility,
            )
            _, indices = exact_slot_oracle(env)
            choices.add(indices)

        unique_counts.append(len(choices))
        changed_worlds += int(len(choices) > 1)

    return {
        "changed_world_fraction": changed_worlds / worlds,
        "mean_unique_argmax": float(np.mean(unique_counts)),
        "max_unique_argmax": float(np.max(unique_counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=25)
    parser.add_argument("--personas-per-world", type=int, default=4)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    base = EnvironmentConfig.from_yaml(args.config)
    config = replace(
        base,
        price_drift_per_step=0.0,
        depletion_probability=0.0,
        disruption_probability=0.0,
    )
    results = evaluate_persona_sensitivity(
        config,
        worlds=args.worlds,
        personas_per_world=args.personas_per_world,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    for metric, value in results.items():
        print(f"{metric:<24} {value:.4f}")
    print(
        "\nEach world keeps inventory, request, and hard-budget flexibility "
        "fixed while resampling latent utility preferences."
    )


if __name__ == "__main__":
    main()
