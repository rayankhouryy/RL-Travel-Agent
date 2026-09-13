from travel_env import TravelAgentEnv
from travel_env.actions import ActionType
from travel_env.config import EnvironmentConfig


def action(action_type: ActionType, target: int = 0, source: int = 0):
    return {
        "action_type": int(action_type),
        "target_index": target,
        "source_index": source,
    }


def test_flight_cancellation_invalidates_early_booked_activities():
    env = TravelAgentEnv(EnvironmentConfig(disruption_probability=0.0))
    env.reset(seed=44)
    state = env.state

    flight = next(item for item in state.inventory if item.category.value == 0)
    activity = next(
        item
        for item in state.inventory
        if item.category.value == 2
        and item.start_hour < flight.end_hour + 12.0
    )
    state.booked.update({flight.index, activity.index})
    state.spent = flight.price + activity.price

    env.disruption_engine.config = EnvironmentConfig(
        disruption_probability=1.0
    )
    disruption = env.disruption_engine.maybe_trigger(
        state,
        _ChooseFlightRng(flight.index),
    )

    assert disruption.kind == "flight_cancellation"
    assert activity.index in disruption.affected_indices
    assert not flight.available
    assert not activity.available
    assert {flight.index, activity.index}.issubset(
        state.active_disruption_targets()
    )


class _ChooseFlightRng:
    def __init__(self, target):
        self.target = target

    def random(self):
        return 0.0

    def choice(self, candidates):
        assert self.target in candidates
        return self.target

