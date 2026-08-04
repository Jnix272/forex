"""
risk/portfolio_allocator.py
===========================
Dynamically calculates optimal position sizing (lot sizes) based on AI confidence,
win rate, and market volatility using the Kelly Criterion and Risk Parity algorithms.
"""

import logging


class PortfolioAllocator:
    def __init__(self, max_risk_per_trade: float = 0.02, account_balance: float = 10000.0):
        self.logger = logging.getLogger(__name__)
        self.max_risk = max_risk_per_trade  # e.g., max 2% of account per trade
        self.balance = account_balance

    def update_balance(self, new_balance: float):
        self.balance = new_balance

    def calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate the Kelly Criterion fraction.
        f* = W - ((1 - W) / R)
        W: Historical win probability (0 to 1)
        R: Reward to Risk ratio (average win size / average loss size)
        """
        if win_loss_ratio <= 0:
            return 0.0

        kelly = win_rate - ((1.0 - win_rate) / win_loss_ratio)
        return max(0.0, kelly) # Never return negative sizing

    def get_optimal_lot_size(
        self,
        model_confidence: float,
        win_rate: float,
        win_loss_ratio: float,
        stop_loss_pips: float,
        pip_value: float = 10.0,
        fractional_kelly: float = 0.5
    ) -> float:
        """
        Calculate the exact lot size to trade based on Half-Kelly criterion,
        capped by the maximum risk parameter.
        """
        if stop_loss_pips <= 0:
            self.logger.warning("Stop loss must be > 0. Returning 0 lots.")
            return 0.0

        # 1. Calculate Kelly
        raw_kelly = self.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # We usually trade "Half-Kelly" to reduce drawdowns
        safe_kelly = raw_kelly * fractional_kelly

        # Modify by real-time model confidence (if the model is 50% confident, halve the size)
        adjusted_kelly = safe_kelly * model_confidence

        # Cap the Kelly fraction to our strict maximum risk parameter (e.g. 2%)
        final_risk_fraction = min(adjusted_kelly, self.max_risk)

        # 2. Translate Risk Fraction to Dollar Amount
        risk_dollars = self.balance * final_risk_fraction

        # 3. Translate Dollar Risk to Lot Size based on Stop Loss distance
        # Risk = Lots * StopLossPips * PipValue
        # Lots = Risk / (StopLossPips * PipValue)

        lots = risk_dollars / (stop_loss_pips * pip_value)

        # Round to nearest micro lot (0.01)
        return round(lots, 2)
