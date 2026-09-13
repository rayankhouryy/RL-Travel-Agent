from evaluation.evaluate import evaluate
from travel_env.config import EnvironmentConfig


def test_heuristic_materially_outperforms_random_policy():
    config = EnvironmentConfig(disruption_probability=0.04)

    random = evaluate("random", episodes=30, seed=800, config=config)
    heuristic = evaluate("heuristic", episodes=30, seed=800, config=config)

    assert heuristic["success"] >= random["success"] + 0.50
    assert heuristic["reward"] > random["reward"]
    assert heuristic["satisfaction"] > random["satisfaction"]
