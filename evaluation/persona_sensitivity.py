from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, Iterable

import numpy as np

from evaluation.oracle_regret import exact_slot_oracle
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig


def itinerary_utility(
    env: TravelAgentEnv,
    indices: Iterable[int],
) -> float:
    state = env.state
    original_booked = set(state.booked)
    original_spent = state.spent
    state.booked = set(indices)
    state.spent = sum(state.inventory[index].price for index in indices)
    utility = env.reward_model.latent_utility(state)
    state.booked = original_booked
    state.spent = original_spent
    return utility


def evaluate_persona_sensitivity(
    config: EnvironmentConfig,
    worlds: int,
    personas_per_world: int,
    seed: int,
    difficulty: int,
) -> Dict[str, float]:
    changed_worlds = 0
    unique_counts = []
    cross_persona_losses = []
    optimal_utilities = []

    for world_index in range(worlds):
        world_seed = seed + world_index
        env = TravelAgentEnv(config)
        env.reset(seed=world_seed, options={"difficulty": difficulty})
        base_flexibility = env.state.persona.budget_flexibility
        personas = []
        choices = []
        utilities = []

        for persona_index in range(personas_per_world):
            donor = TravelAgentEnv(config)
            donor.reset(
                seed=seed + 100000 + world_index * personas_per_world
                + persona_index,
                options={"difficulty": difficulty},
            )
            persona = replace(
                donor.state.persona,
                budget_flexibility=base_flexibility,
            )
            personas.append(persona)
            env.state.persona = persona
            utility, indices = exact_slot_oracle(env)
            utilities.append(utility)
            choices.append(indices)

        unique_count = len(set(choices))
        unique_counts.append(unique_count)
        changed_worlds += int(unique_count > 1)
        optimal_utilities.extend(utilities)

        for target_index, persona in enumerate(personas):
            env.state.persona = persona
            target_optimum = utilities[target_index]
            for source_index, source_choice in enumerate(choices):
                if source_index == target_index:
                    continue
                transferred = itinerary_utility(env, source_choice)
                loss = target_optimum - transferred
                if loss < -1e-9:
                    raise AssertionError(
                        "Transferred itinerary exceeded exact target optimum"
                    )
                cross_persona_losses.append(max(loss, 0.0))

    return {
        "changed_world_fraction": changed_worlds / worlds,
        "mean_unique_argmax": float(np.mean(unique_counts)),
        "max_unique_argmax": float(np.max(unique_counts)),
        "mean_cross_persona_loss": float(
            np.mean(cross_persona_losses)
        ),
        "median_cross_persona_loss": float(
            np.median(cross_persona_losses)
        ),
        "p90_cross_persona_loss": float(
            np.quantile(cross_persona_losses, 0.90)
        ),
        "relative_cross_persona_loss": float(
            np.mean(cross_persona_losses)
            / max(float(np.mean(optimal_utilities)), 1e-8)
        ),
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
