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
    CLOSE = 3
    SCALE_IN_25 = 4
    SCALE_IN_50 = 5
    SCALE_IN_100 = 6
    SCALE_OUT_25 = 7
    SCALE_OUT_50 = 8
    SCALE_OUT_100 = 9


MODEL_CLASS_TO_LIVE_ACTION = {
    0: LiveAction.SELL,
    1: LiveAction.HOLD,
    2: LiveAction.BUY,
}


def model_class_to_live_action(action: int) -> int:
    """Translate supervised model class 0/1/2 into the live action contract."""
    return int(MODEL_CLASS_TO_LIVE_ACTION.get(int(action), LiveAction.HOLD))


def scaling_action_to_live_action(action: int, position_lots: float = 0.0) -> int:
    """Translate RL ScalingAction values into live actions.

    Supports the full 10-action ScalingAction space:
      0: HOLD
      1: OPEN_LONG
      2: OPEN_SHORT
      3: SCALE_IN_25
      4: SCALE_IN_50
      5: SCALE_IN_100
      6: SCALE_OUT_25
      7: SCALE_OUT_50
      8: SCALE_OUT_100
      9: CLOSE_ALL
    """
    action = int(action)
    if action == 1:  # OPEN_LONG
        return int(LiveAction.BUY)
    if action == 2:  # OPEN_SHORT
        return int(LiveAction.SELL)
    if action in (3, 4, 5):  # SCALE_IN_*
        if abs(position_lots) < 1e-6:
            return int(LiveAction.HOLD)
        if action == 3:
            return int(LiveAction.SCALE_IN_25)
        if action == 4:
            return int(LiveAction.SCALE_IN_50)
        return int(LiveAction.SCALE_IN_100)
    if action in (6, 7, 8):  # SCALE_OUT_*
        if abs(position_lots) < 1e-6:
            return int(LiveAction.HOLD)
        if action == 6:
            return int(LiveAction.SCALE_OUT_25)
        if action == 7:
            return int(LiveAction.SCALE_OUT_50)
        return int(LiveAction.SCALE_OUT_100)
    if action == 9:  # CLOSE_ALL
        if abs(position_lots) < 1e-6:
            return int(LiveAction.HOLD)
        return int(LiveAction.CLOSE)
    return int(LiveAction.HOLD)


def scaling_action_to_simple_action(action: int, position_lots: float = 0.0) -> int:
    """Translate RL ScalingAction into 3-action space (0=Buy, 1=Hold, 2=Sell) for legacy callers.

    Critically, CLOSE and SCALE_OUT actions are mapped to opposite-side orders to actually
    reduce/flatten exposure instead of silently suppressing exits via HOLD.
    """
    action = int(action)
    if action == 1:
        return int(LiveAction.BUY)
    if action == 2:
        return int(LiveAction.SELL)
    if action in (3, 4, 5):
        if abs(position_lots) < 1e-6:
            return int(LiveAction.HOLD)
        return int(LiveAction.SELL if position_lots < 0 else LiveAction.BUY)
    if action in (6, 7, 8, 9):
        if abs(position_lots) < 1e-6:
            return int(LiveAction.HOLD)
        # To reduce or close a long, sell; to reduce or close a short, buy.
        return int(LiveAction.BUY if position_lots < 0 else LiveAction.SELL)
    return int(LiveAction.HOLD)

