from __future__ import annotations

from typing import Dict

import numpy as np

from travel_env.actions import ActionType, TARGETED_ACTIONS


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, int]:
        valid_actions = np.flatnonzero(observation["action_mask"])
        action_type = ActionType(int(self.rng.choice(valid_actions)))
        target = 0
        source = 0

        if action_type in TARGETED_ACTIONS:
            valid_targets = np.flatnonzero(
                observation["target_mask"][int(action_type)]
            )
            if len(valid_targets):
                target = int(self.rng.choice(valid_targets))

        if action_type in (ActionType.SWAP_BOOKING, ActionType.REBOOK):
            booked = np.flatnonzero(observation["booking_mask"])
            disruptions = np.flatnonzero(observation["disruption_mask"])
            choices = (
                disruptions
                if action_type == ActionType.REBOOK and len(disruptions)
                else booked
            )
            if len(choices):
                source = int(self.rng.choice(choices))

        return {
            "action_type": int(action_type),
            "target_index": target,
            "source_index": source,
        }

