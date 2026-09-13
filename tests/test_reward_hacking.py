from evaluation.reward_hacking import (
    evaluate_policies,
    fragility_probe,
)
from travel_env.config import EnvironmentConfig


def test_exploit_policies_produce_distinct_behavior():
    config = EnvironmentConfig(disruption_probability=0.08)

    results = evaluate_policies(
        config,
        episodes=40,
        seed=14000,
        difficulty=3,
    )

    assert (
        results["quality_maximizer"]["quality"]
        > results["cheapest"]["quality"]
    )
    assert (
        results["flexibility_buyer"]["refundable_share"]
        > results["nonrefundable"]["refundable_share"] + 0.40
    )
    assert results["heuristic"]["reward"] > results["random"]["reward"]


def test_paired_flexibility_probe_detects_fragility_gap():
    config = EnvironmentConfig(disruption_probability=0.08)

    probe = fragility_probe(
        config,
        episodes=100,
        seed=15000,
        difficulty=3,
    )

    assert probe["refundable_share"][0] > 0.40
    assert probe["sunk_fraction"][0] < 0.0
    assert probe["realized"][0] > 0.0
