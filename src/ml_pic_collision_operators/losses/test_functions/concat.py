import torch

from ml_pic_collision_operators.losses.test_functions.base import TestFunction


class ConcatTestFunctions(TestFunction):
    """Composite that concatenates F test-function families."""

    def __init__(self, components: list[TestFunction]):
        if not components:
            raise ValueError("ConcatTestFunctions requires at least one component")
        n_dims = components[0].n_dims
        for c in components[1:]:
            if c.n_dims != n_dims:
                raise ValueError(
                    f"All components must share n_dims; got {n_dims} vs {c.n_dims}"
                )
        self.components = list(components)
        self.n_dims = int(n_dims)
        self.n_functions = sum(c.n_functions for c in components)

    def evaluate_phi(self, x: torch.Tensor) -> torch.Tensor:
        # (B,N,F)
        return torch.cat([c.evaluate_phi(x) for c in self.components], dim=-1)

    def evaluate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evals = [c.evaluate(x) for c in self.components]
        # (B,N,F)
        phi = torch.cat([e[0] for e in evals], dim=-1)
        # (B,N,F,D)
        grad = torch.cat([e[1] for e in evals], dim=-2)
        # (B,N,F,D,D)
        hess = torch.cat([e[2] for e in evals], dim=-3)
        return phi, grad, hess
