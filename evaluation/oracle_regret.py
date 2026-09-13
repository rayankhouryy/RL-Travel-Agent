from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from typing import Dict, Iterable, Tuple

import numpy as np

from policies import HeuristicPolicy
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig
from travel_env.models import InventoryCategory, InventoryItem, TravelState


def compatible(activities: Iterable[InventoryItem]) -> bool:
    ordered = sorted(activities, key=lambda item: item.start_hour)
    return all(
        current.end_hour <= following.start_hour
        for current, following in zip(ordered, ordered[1:])
    )


def exact_slot_oracle(env: TravelAgentEnv) -> Tuple[float, Tuple[int, ...]]:
    state = env.state
    flights = [
        item
        for item in state.inventory
        if item.category == InventoryCategory.FLIGHT and item.available
    ]
    hotels = [
        item
        for item in state.inventory
        if item.category == InventoryCategory.HOTEL and item.available
    ]
    activities = [
        item
        for item in state.inventory
        if item.category == InventoryCategory.ACTIVITY and item.available
    ]

    original_booked = set(state.booked)
    original_spent = state.spent
    best_utility = -1.0
    best_indices: Tuple[int, ...] = ()
    for flight in flights:
        for hotel in hotels:
            base_cost = flight.price + hotel.price
            if base_cost > state.hard_budget():
                continue
            for selected in itertools.combinations(
                activities,
                state.required_activities,
            ):
                if not compatible(selected):
                    continue
                cost = base_cost + sum(item.price for item in selected)
                if cost > state.hard_budget():
                    continue
                state.booked = {
                    flight.index,
                    hotel.index,
                    *(item.index for item in selected),
                }
                state.spent = cost
                utility = env.reward_model.latent_utility(state)
                if utility > best_utility:
                    best_utility = utility
                    best_indices = tuple(sorted(state.booked))

    state.booked = original_booked
    state.spent = original_spent
    if best_utility < 0:
        raise RuntimeError("No feasible itinerary found for oracle evaluation")
    return best_utility, best_indices


def heuristic_utility(
    config: EnvironmentConfig, seed: int, difficulty: int
) -> Tuple[float, bool]:
    env = TravelAgentEnv(config)
    policy = HeuristicPolicy()
    observation, _ = env.reset(seed=seed, options={"difficulty": difficulty})
    terminated = truncated = False
    info: Dict = {}
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(
            policy.act(observation)
        )
    return env.reward_model.latent_utility(env.state), bool(
        terminated and info["client_accepted"]
    )


def evaluate_regret(
    config: EnvironmentConfig,
    episodes: int,
    seed: int,
    difficulty: int,
) -> Dict[str, float]:
    oracle_values = []
    agent_values = []
    successes = []
    for episode in range(episodes):
        episode_seed = seed + episode
        oracle_env = TravelAgentEnv(config)
        oracle_env.reset(
            seed=episode_seed,
            options={"difficulty": difficulty},
        )
        oracle_value, _ = exact_slot_oracle(oracle_env)
        agent_value, success = heuristic_utility(
            config,
            episode_seed,
            difficulty,
        )
        oracle_values.append(oracle_value)
        agent_values.append(agent_value)
        successes.append(float(success))

    regrets = np.asarray(oracle_values) - np.asarray(agent_values)
    success_mask = np.asarray(successes, dtype=bool)
    failure_mask = ~success_mask
    return {
        "oracle_utility": float(np.mean(oracle_values)),
        "agent_utility": float(np.mean(agent_values)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "p90_regret": float(np.quantile(regrets, 0.90)),
        "successful_mean_regret": (
            float(np.mean(regrets[success_mask]))
            if np.any(success_mask)
            else float("nan")
        ),
        "unsuccessful_mean_regret": (
            float(np.mean(regrets[failure_mask]))
            if np.any(failure_mask)
            else float("nan")
        ),
        "agent_success": float(np.mean(successes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=40000)
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
    results = evaluate_regret(
        config,
        episodes=args.episodes,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    for metric, value in results.items():
        print(f"{metric:<20} {value:.4f}")
    print(
        "\nOracle scope: exact over one flight, one hotel, and exactly the "
        "required number of non-overlapping activities in a static, "
        "fully observed, no-disruption world."
    )


if __name__ == "__main__":
    main()
