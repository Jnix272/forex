import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """
    Unified Exponential Moving Average (EMA) utility.

    Supports:
    - Scalar values (e.g. tracking Sharpe ratios or loss)
    - PyTorch modules (e.g. target networks in BYOL / self-supervised learning)
    """

    def __init__(self, alpha: float):
        """
        Args:
            alpha (float): The decay rate (0 < alpha < 1).
                           For scalars: new_val = alpha * current + (1 - alpha) * previous
                           For modules: new_weight = alpha * target_weight + (1 - alpha) * online_weight
        """
        self.alpha = float(alpha)
        self.value: float | torch.Tensor | None = None

    def update_scalar(self, current: float) -> float:
        """
        Update the EMA for a scalar value.

        Args:
            current (float): The new scalar measurement.
        Returns:
            float: The updated EMA scalar.
        """
        if self.value is None:
            self.value = float(current)
        else:
            self.value = self.alpha * float(current) + (1.0 - self.alpha) * self.value
        return self.value

    @staticmethod
    def update_module(online_network: nn.Module, target_network: nn.Module, alpha: float) -> None:
        """
        Update a target PyTorch module's parameters using an EMA of an online module.
        Typically used in BYOL or contrastive learning.

        target_weight = alpha * target_weight + (1 - alpha) * online_weight

        Args:
            online_network: The actively training network (source of new weights)
            target_network: The EMA target network (receives the update)
            alpha: Decay rate
        """
        with torch.no_grad():
            for param_q, param_k in zip(online_network.parameters(), target_network.parameters(), strict=False):
                param_k.data = param_k.data * alpha + param_q.data * (1.0 - alpha)
