import torch
import numpy as np

from ml_pic_collision_operators.losses.test_functions.base import TestFunction


class RadialQuadraticGaussianTestFunctions(TestFunction):
    """Squared-radius times isotropic Gaussian, at fixed centers.

    For each center μ_f ∈ R^D and standard deviation σ_f > 0, let
    δ = x - μ_f, r² = ‖δ‖², and g = exp(-r² / (2 σ_f²)). Then

        φ_f(x) = r² · g

        ∂_j φ_f(x) = δ_j · g · (2 - r² / σ_f²)

        ∂_i ∂_j φ_f(x) = g · [ δ_{ij} · (2 - r² / σ_f²)
                               + (δ_i δ_j / σ_f²) · (r² / σ_f² - 4) ]

    Args:
        centers: (F, D) array-like of test-function centers in the same
            coordinates as the input `x` passed to `evaluate`.
        sigma: Bandwidth(s). Either a scalar (shared across all F functions)
            or a length-F array-like (per-function standard deviation).
    """

    def __init__(self, centers: list | np.ndarray, sigma: float | list | np.ndarray):
        centers_np = np.asarray(centers, dtype=np.float64)
        if centers_np.ndim != 2:
            raise ValueError(
                f"centers must be a (F, D) matrix, got array with shape {centers_np.shape}"
            )
        self.n_functions = int(centers_np.shape[0])
        self.n_dims = int(centers_np.shape[1])

        sigma_np = np.asarray(sigma, dtype=np.float64)
        if sigma_np.ndim == 0:
            sigma_np = np.full((self.n_functions,), float(sigma_np))
        elif sigma_np.shape != (self.n_functions,):
            raise ValueError(
                f"sigma must be scalar or shape ({self.n_functions},), got {sigma_np.shape}"
            )
        if np.any(sigma_np <= 0):
            raise ValueError("sigma values must be strictly positive")

        self.centers = centers_np
        self.sigma = sigma_np
        self._cache: dict[
            tuple[torch.device, torch.dtype],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _params(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = (
                torch.as_tensor(self.centers, device=device, dtype=dtype),
                torch.as_tensor(self.sigma, device=device, dtype=dtype),
            )
        return self._cache[key]

    def evaluate_phi(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_dims:
            raise ValueError(f"x last dim {x.shape[-1]} != n_dims {self.n_dims}")

        D = x.shape[-1]
        F = self.n_functions
        centers, sigma = self._params(x.device, x.dtype)

        # (B,N,F,D)
        delta = x.unsqueeze(2) - centers.view(1, 1, F, D)
        # (B,N,F)
        r2 = (delta * delta).sum(dim=-1)
        # (1,1,F)
        inv_s2 = (1.0 / (sigma**2)).view(1, 1, F)
        # phi = r^2 * exp(-r^2 / (2 sigma^2))
        return r2 * torch.exp(-0.5 * r2 * inv_s2)

    def evaluate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.n_dims:
            raise ValueError(f"x last dim {x.shape[-1]} != n_dims {self.n_dims}")

        B, N, D = x.shape
        F = self.n_functions
        centers, sigma = self._params(x.device, x.dtype)

        # x - mu
        # (B,N,F,D)
        delta = x.unsqueeze(2) - centers.view(1, 1, F, D)
        # r^2 = (x - mu)^2
        # (B,N,F)
        r2 = (delta * delta).sum(dim=-1)

        # (1,1,F)
        inv_s2 = (1.0 / (sigma**2)).view(1, 1, F)
        # (B,N,F)
        g = torch.exp(-0.5 * r2 * inv_s2)

        # Test function values
        # (B,N,F)
        phi = r2 * g

        # Test function gradients
        # (B,N,F)
        scale_grad = g * (2.0 - r2 * inv_s2)
        # (B,N,F,D)
        grad = delta * scale_grad.unsqueeze(-1)

        # Test function hessian
        # (B,N,F,D,D)
        delta_outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)
        # (1,1,1,D,D)
        eye = torch.eye(D, device=x.device, dtype=x.dtype).view(1, 1, 1, D, D)
        # (B,N,F,1,1)
        diag_coef = (g * (2.0 - r2 * inv_s2)).unsqueeze(-1).unsqueeze(-1)
        outer_coef = (g * inv_s2 * (r2 * inv_s2 - 4.0)).unsqueeze(-1).unsqueeze(-1)
        # (B,N,F,D,D)
        hess = eye * diag_coef + delta_outer * outer_coef

        return phi, grad, hess
