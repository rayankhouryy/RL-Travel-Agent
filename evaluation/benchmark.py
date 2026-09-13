from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
import tracemalloc
from dataclasses import replace
from typing import Dict, Iterable, List

from policies import HeuristicPolicy
from travel_env import TravelAgentEnv
from travel_env.config import EnvironmentConfig


def benchmark_resets(
    config: EnvironmentConfig, resets: int, seed: int
) -> float:
    env = TravelAgentEnv(config)
    started = time.perf_counter()
    for index in range(resets):
        env.reset(seed=seed + index, options={"difficulty": 2})
    elapsed = time.perf_counter() - started
    return 1000.0 * elapsed / resets


def benchmark_steps(
    config: EnvironmentConfig,
    env_count: int,
    total_steps: int,
    seed: int,
) -> Dict[str, float]:
    envs = [TravelAgentEnv(config) for _ in range(env_count)]
    policies = [HeuristicPolicy() for _ in range(env_count)]
    observations = [
        env.reset(seed=seed + index, options={"difficulty": 2})[0]
        for index, env in enumerate(envs)
    ]
    episodes = 0
    started = time.perf_counter()

    for step in range(total_steps):
        index = step % env_count
        env = envs[index]
        observation, _, terminated, truncated, _ = env.step(
            policies[index].act(observations[index])
        )
        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset(
                seed=seed + env_count + episodes,
                options={"difficulty": 2},
            )
        observations[index] = observation

    elapsed = time.perf_counter() - started
    return {
        "steps_per_second": total_steps / elapsed,
        "episodes_per_second": episodes / elapsed,
        "elapsed_seconds": elapsed,
    }


def memory_per_environment(
    config: EnvironmentConfig, env_count: int, seed: int
) -> float:
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    envs = [TravelAgentEnv(config) for _ in range(env_count)]
    for index, env in enumerate(envs):
        env.reset(seed=seed + index, options={"difficulty": 2})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del envs
    return max(0.0, peak - before) / env_count / 1024.0


def profile_steps(
    config: EnvironmentConfig, steps: int, seed: int
) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    benchmark_steps(config, env_count=1, total_steps=steps, seed=seed)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
    return stream.getvalue()


def parse_ints(values: Iterable[str]) -> List[int]:
    return [int(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--resets", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument(
        "--env-counts",
        nargs="+",
        default=["1", "8", "32"],
    )
    parser.add_argument(
        "--inventory-sizes",
        nargs="+",
        default=["24", "32", "64"],
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    base = EnvironmentConfig.from_yaml(args.config)
    env_counts = parse_ints(args.env_counts)
    inventory_sizes = parse_ints(args.inventory_sizes)

    print("Inventory scaling")
    print(
        f"{'inventory':>10} {'reset_ms':>12} "
        f"{'steps/s':>14} {'KiB/env':>12}"
    )
    for size in inventory_sizes:
        config = replace(base, max_inventory=size)
        reset_ms = benchmark_resets(config, args.resets, args.seed)
        throughput = benchmark_steps(
            config,
            env_count=1,
            total_steps=args.steps,
            seed=args.seed,
        )
        memory = memory_per_environment(config, 32, args.seed)
        print(
            f"{size:>10d} {reset_ms:>12.3f} "
            f"{throughput['steps_per_second']:>14.1f} {memory:>12.1f}"
        )

    print("\nEnvironment-count scaling")
    print(f"{'envs':>10} {'steps/s':>14} {'episodes/s':>14} {'KiB/env':>12}")
    for count in env_counts:
        throughput = benchmark_steps(
            base,
            env_count=count,
            total_steps=args.steps,
            seed=args.seed,
        )
        memory = memory_per_environment(base, count, args.seed)
        print(
            f"{count:>10d} {throughput['steps_per_second']:>14.1f} "
            f"{throughput['episodes_per_second']:>14.2f} {memory:>12.1f}"
        )

    if args.profile:
        print("\nProfile")
        print(profile_steps(base, args.steps, args.seed))


if __name__ == "__main__":
    main()

