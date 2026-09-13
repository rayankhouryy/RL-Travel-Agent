from evaluation.recovery_analysis import evaluate_recovery
from travel_env.config import EnvironmentConfig


def test_recovery_analysis_reports_both_success_groups():
    result = evaluate_recovery(
        EnvironmentConfig(disruption_probability=0.08),
        episodes=80,
        seed=90000,
        difficulty=2,
    )

    assert result["no_disruption"]["episodes"] > 0
    assert result["recovered_disruption"]["episodes"] > 0
    assert result["no_disruption"]["terminal_utility_std"] >= 0.0
    assert result["recovered_disruption"]["terminal_utility_std"] >= 0.0
