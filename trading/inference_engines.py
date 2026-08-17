import abc

import numpy as np


class BaseInferenceEngine(abc.ABC):
    """
    Abstract base class for all inference engines used in the live trading pipeline.
    Enforces a consistent interface across PyTorch, ONNX/DirectML, and RL agents.
    """

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """
        Takes a single bar's feature vector (1D array), updates internal rolling
        buffers or state, and returns an action.

        Returns:
            int: 0=Buy, 1=Hold, 2=Sell
        """
        pass

    @abc.abstractmethod
    def reset_buffer(self) -> None:
        """
        Clears any internal rolling observation buffers or recurrent state.
        Called on session restarts or large time gaps.
        """
        pass
