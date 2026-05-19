import torch
from abc import ABC, abstractmethod


class TestFunction(ABC):
    """Closed-form test function family for weak-form SDE losses.

    Subclasses evaluate a fixed set of F scalar test functions plus their
    first and second derivatives at arbitrary particle positions.
    """

    n_dims: int
    n_functions: int

    @abstractmethod
    def evaluate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate values, gradients, Hessians of all test functions at x.

        Args:
            x: particle positions with shape (B, N, D).

        Returns:
            phi:  (B, N, F)
            grad: (B, N, F, D)
            hess: (B, N, F, D, D)
        """
        ...

    def evaluate_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate values only (no gradients, no Hessians).

        Default falls back to `evaluate` and discards grad/hess. Subclasses
        should override to skip the unused derivative work when it is
        expensive (rollout-style weak losses only need phi).

        Args:
            x: particle positions with shape (B, N, D).

        Returns:
            phi: (B, N, F)
        """
        phi, _, _ = self.evaluate(x)
        return phi
