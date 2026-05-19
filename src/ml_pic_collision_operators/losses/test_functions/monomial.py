import itertools
import torch
import numpy as np

from ml_pic_collision_operators.losses.test_functions.base import TestFunction


class MonomialTestFunctions(TestFunction):
    """Multivariate monomials of total degree `1` to `degree` over D dimensions.

    For each multi-index α = (α_1, ..., α_D) with 1 ≤ |α| ≤ degree,

        φ_α(x) = ∏_d x_d^{α_d}.

    The constant function (|α| = 0) is excluded.

    Args:
        n_dims: Number of spatial dimensions D.
        degree: Maximum total degree of the monomials. The resulting family has
            F = C(D + degree, D) - 1 functions (all multi-indices of total degree 1
            to `degree`).
    """

    def __init__(self, n_dims: int, degree: int):
        if n_dims < 1:
            raise ValueError(f"n_dims must be ≥ 1, got {n_dims}")
        if degree < 1:
            raise ValueError(f"degree must be ≥ 1, got {degree}")
        self.n_dims = int(n_dims)
        self.degree = int(degree)

        alpha_list: list[tuple[int, ...]] = []
        for degree in range(1, self.degree + 1):
            for alpha in itertools.product(range(degree + 1), repeat=self.n_dims):
                if sum(alpha) == degree:
                    alpha_list.append(alpha)
        self.alpha = np.asarray(alpha_list, dtype=np.int64)
        self.n_functions = self.alpha.shape[0]
        self._alpha_cache: dict[torch.device, torch.Tensor] = {}

    def _alpha(self, device: torch.device) -> torch.Tensor:
        if device not in self._alpha_cache:
            self._alpha_cache[device] = torch.from_numpy(self.alpha).to(device)
        return self._alpha_cache[device]

    def _power_table(self, x: torch.Tensor) -> torch.Tensor:
        exponents = torch.arange(self.degree + 1, device=x.device, dtype=x.dtype)
        # Power table: pw[..., d, p] = x[..., d]^p for p = 0..degree
        # (B, N, D, degree+1)
        return x.unsqueeze(-1) ** exponents

    def evaluate_phi(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_dims:
            raise ValueError(f"x last dim {x.shape[-1]} != n_dims {self.n_dims}")
        D = x.shape[-1]
        alpha = self._alpha(x.device)
        pw = self._power_table(x)
        terms = [pw[..., d, :].index_select(-1, alpha[:, d]) for d in range(D)]
        return torch.stack(terms, dim=-1).prod(dim=-1)

    def evaluate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.n_dims:
            raise ValueError(f"x last dim {x.shape[-1]} != n_dims {self.n_dims}")

        B, N, D = x.shape
        F = self.n_functions
        # (F, D)
        alpha = self._alpha(x.device)
        pw = self._power_table(x)

        def _prod_with_shift(shift_j: int | None, shift_k: int | None) -> torch.Tensor:
            """Computes ∏_d x_d^{α__{k,d} - δ_{d,j} - δ_{d,k}}."""

            shifted = alpha.clone()
            if shift_j is not None:
                shifted[:, shift_j] -= 1
            if shift_k is not None:
                shifted[:, shift_k] -= 1
            # Prefactors are 0 when alpha was 0
            shifted = shifted.clamp(min=0)
            # Gather pw[..., d, shifted[:, d]] for each d, then product over d.
            terms = []
            for d in range(D):
                # pw[..., d, :] has shape (B, N, degree+1)
                # index with shifted[:, d] (F,)
                terms.append(pw[..., d, :].index_select(-1, shifted[:, d]))
            # (B, N, F, D)
            stacked = torch.stack(terms, dim=-1)
            # (B, N, F)
            return stacked.prod(dim=-1)

        # Test function values
        # (B, N, F)
        phi = _prod_with_shift(None, None)

        # Test function gradients
        grad = x.new_zeros(B, N, F, D)
        for j in range(D):
            # (B, N, F)
            partial = _prod_with_shift(j, None)
            # (F,)
            prefactor = alpha[:, j].to(x.dtype)
            grad[..., j] = prefactor * partial

        # Test function hessian
        hess = x.new_zeros(B, N, F, D, D)
        for j in range(D):
            for k in range(D):
                partial = _prod_with_shift(j, k)
                if j == k:
                    prefactor = (alpha[:, j] * (alpha[:, j] - 1)).to(x.dtype)
                else:
                    prefactor = (alpha[:, j] * alpha[:, k]).to(x.dtype)
                hess[..., j, k] = prefactor * partial

        return phi, grad, hess
