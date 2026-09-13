from __future__ import annotations

from enum import IntEnum
from typing import Mapping


class ActionType(IntEnum):
    SEARCH_FLIGHTS = 0
    SEARCH_HOTELS = 1
    SEARCH_ACTIVITIES = 2
    SELECT_FLIGHT = 3
    SELECT_HOTEL = 4
    SELECT_ACTIVITY = 5
    REMOVE_BOOKING = 6
    SWAP_BOOKING = 7
    PROPOSE_ITINERARY = 8
    MESSAGE_CLIENT = 9
    REBOOK = 10
    FINISH = 11
    ADVANCE_TRIP = 12


TARGETED_ACTIONS = {
    ActionType.SELECT_FLIGHT,
    ActionType.SELECT_HOTEL,
    ActionType.SELECT_ACTIVITY,
    ActionType.REMOVE_BOOKING,
    ActionType.SWAP_BOOKING,
    ActionType.REBOOK,
}


def decode_action(action: Mapping[str, int]) -> tuple:
    try:
        action_type = ActionType(int(action["action_type"]))
        target_index = int(action["target_index"])
        source_index = int(action.get("source_index", target_index))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Action must contain integer action_type, target_index, and source_index"
        ) from exc
    return action_type, target_index, source_index
