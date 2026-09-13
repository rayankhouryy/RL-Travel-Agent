import pytest

from travel_env import TravelAgentEnv
from travel_env.actions import ActionType
from travel_env.config import EnvironmentConfig
from travel_env.models import InventoryCategory


def action(action_type: ActionType, target: int = 0, source: int = 0):
    return {
        "action_type": int(action_type),
        "target_index": target,
        "source_index": source,
    }


def test_unbooked_prices_drift_while_booked_quote_stays_fixed():
    env = TravelAgentEnv(
        EnvironmentConfig(
            price_drift_per_step=0.10,
            depletion_probability=0.0,
        )
    )
    env.reset(seed=410)
    env.step(action(ActionType.SEARCH_FLIGHTS))
    state = env.state
    targets = [
        item
        for item in state.inventory
        if item.category == InventoryCategory.FLIGHT and item.available
    ]
    booked, unbooked = targets[:2]
    env.step(action(ActionType.SELECT_FLIGHT, booked.index))
    quoted_price = state.booking_prices[booked.index]
    unbooked_price = unbooked.price

    env.step(action(ActionType.SEARCH_HOTELS))

    assert state.booking_prices[booked.index] == quoted_price
    assert booked.price == quoted_price
    assert unbooked.price > unbooked_price


def test_refund_uses_booked_quote_not_later_market_price():
    config = EnvironmentConfig(
        price_drift_per_step=0.0,
        depletion_probability=0.0,
        cancellation_fee_rate=0.20,
    )
    env = TravelAgentEnv(config)
    env.reset(seed=411)
    state = env.state
    item = next(
        item
        for item in state.inventory
        if item.available and not item.refundable
    )
    state.booked.add(item.index)
    state.booking_prices[item.index] = item.price
    state.spent = item.price
    quoted_price = item.price
    item.price *= 2.0

    env.step(action(ActionType.REMOVE_BOOKING, target=item.index))

    assert state.spent == pytest.approx(quoted_price * 0.20)
    assert state.sunk_cost == pytest.approx(quoted_price * 0.20)


def test_market_dynamics_are_seed_deterministic():
    config = EnvironmentConfig(
        price_drift_per_step=0.03,
        depletion_probability=0.20,
    )

    def snapshot():
        env = TravelAgentEnv(config)
        env.reset(seed=412, options={"difficulty": 3})
        env.step(action(ActionType.SEARCH_HOTELS))
        env.step(action(ActionType.SEARCH_HOTELS))
        return [
            (item.price, item.available)
            for item in env.state.inventory
        ]

    assert snapshot() == snapshot()
