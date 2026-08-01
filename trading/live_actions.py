"""Action adapters for the live trading boundary.

Model direction logits use class indices 0=sell, 1=hold, 2=buy.  The live
execution loop historically uses 0=buy, 1=hold, 2=sell.  Keep the translation
centralized so inference engines do not silently invert live orders.
"""

from __future__ import annotations

from enum import IntEnum


class LiveAction(IntEnum):
    BUY = 0
    HOLD = 1
    SELL = 2


MODEL_CLASS_TO_LIVE_ACTION = {
    0: LiveAction.SELL,
    1: LiveAction.HOLD,
    2: LiveAction.BUY,
}


def model_class_to_live_action(action: int) -> int:
    """Translate supervised model class 0/1/2 into the live action contract."""
    return int(MODEL_CLASS_TO_LIVE_ACTION.get(int(action), LiveAction.HOLD))


def scaling_action_to_live_action(action: int, position_lots: float = 0.0) -> int:
    """Translate RL ScalingAction values into live buy/hold/sell actions."""
    action = int(action)
    if action == 1:  # OPEN_LONG
        return int(LiveAction.BUY)
    if action == 2:  # OPEN_SHORT
        return int(LiveAction.SELL)
    if action in (3, 4, 5):  # SCALE_IN_* follows the current side if present.
        return int(LiveAction.SELL if position_lots < 0 else LiveAction.BUY)
    return int(LiveAction.HOLD)
