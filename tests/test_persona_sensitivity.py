from dataclasses import replace

from evaluation.persona_sensitivity import evaluate_persona_sensitivity
from travel_env.config import EnvironmentConfig


def test_persona_sensitivity_metrics_are_bounded():
    config = replace(
        EnvironmentConfig(),
        price_drift_per_step=0.0,
        depletion_probability=0.0,
        disruption_probability=0.0,
    )

    result = evaluate_persona_sensitivity(
        config,
        worlds=2,
        personas_per_world=3,
        seed=80000,
        difficulty=1,
    )

    assert 0.0 <= result["changed_world_fraction"] <= 1.0
    assert 1.0 <= result["mean_unique_argmax"] <= 3.0
    assert 1.0 <= result["max_unique_argmax"] <= 3.0
    assert result["mean_cross_persona_loss"] >= 0.0
    assert result["median_cross_persona_loss"] >= 0.0
    assert result["p90_cross_persona_loss"] >= 0.0
    assert result["relative_cross_persona_loss"] >= 0.0
