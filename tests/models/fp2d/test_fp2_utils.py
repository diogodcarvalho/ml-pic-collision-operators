import torch

from ml_pic_collision_operators.models.fp2d.fp2d_utils import (
    _L_decompose,
    fp2d_sde_step,
    fp2d_step,
)


class TestLDecompose:

    def test_recovers_D_for_anisotropic_psd(self):
        # L L^T must equal D exactly for any symmetric PSD D.
        torch.manual_seed(0)
        M = torch.randn(64, 2, 2, dtype=torch.float64)
        D_mat = M @ M.transpose(-1, -2)
        D = torch.stack([D_mat[..., 0, 0], D_mat[..., 1, 1], D_mat[..., 0, 1]], dim=-1)
        L = _L_decompose(D, eps_psd=1e-12)
        assert L.shape == (64, 2, 2)
        assert torch.allclose(L @ L.transpose(-1, -2), D_mat, atol=1e-10)

    def test_recovers_D_for_isotropic(self):
        # Degenerate eigenvalues are the worst case for eigh-based sqrts.
        # The closed-form must still recover D = d·I exactly.
        d = torch.rand(32, dtype=torch.float64) + 0.1
        D = torch.stack([d, d, torch.zeros_like(d)], dim=-1)
        L = _L_decompose(D, eps_psd=1e-12)
        D_mat = torch.zeros(32, 2, 2, dtype=torch.float64)
        D_mat[..., 0, 0] = d
        D_mat[..., 1, 1] = d
        assert torch.allclose(L @ L.transpose(-1, -2), D_mat, atol=1e-10)

    def test_finite_for_indefinite_D(self):
        # The eps_psd floor must keep L finite when det(D) < 0 (which would
        # otherwise produce a negative-radicand sqrt and NaNs).
        D = torch.tensor([[0.1, 0.1, 0.5]], dtype=torch.float64)  # det = -0.24
        L = _L_decompose(D, eps_psd=1e-8)
        assert torch.isfinite(L).all()


class TestFP2DSDEStep:

    def test_deterministic_when_D_is_zero(self):
        # With D = 0 the step must reduce to v + A·dt regardless of the noise sample.
        torch.manual_seed(0)
        B, N = 2, 5
        v = torch.randn(B, N, 2, dtype=torch.float64)
        A = torch.randn(B, N, 2, dtype=torch.float64)
        D = torch.zeros(B, N, 3, dtype=torch.float64)
        dt = 0.1
        v_new = fp2d_sde_step(A, D, v, dt, eps_psd=1e-12)
        assert torch.allclose(v_new, v + A * dt, atol=1e-5)

    def test_dt_scalar_and_tensor_consistent(self):
        # Per-batch tensor dt with equal entries must match scalar dt under the same
        # noise. Needed so variable-dt batches stay consistent with the constant-dt path.
        B, N = 2, 4
        v = torch.randn(B, N, 2)
        A = torch.randn(B, N, 2)
        D = torch.zeros(B, N, 3)
        torch.manual_seed(42)
        v_scalar = fp2d_sde_step(A, D, v, 0.1, eps_psd=1e-12)
        torch.manual_seed(42)
        v_tensor = fp2d_sde_step(A, D, v, torch.full((B,), 0.1), eps_psd=1e-12)
        assert torch.allclose(v_scalar, v_tensor)

    def test_brownian_increment_covariance_matches_D_dt(self):
        # With A = 0, sample covariance of (v_new - v) must match D·dt.
        N = 20000
        dt = 0.05
        Dxx, Dyy, Dxy = 0.4, 0.2, 0.1
        v = torch.zeros(1, N, 2, dtype=torch.float64)
        A = torch.zeros(1, N, 2, dtype=torch.float64)
        D = torch.zeros(1, N, 3, dtype=torch.float64)
        D[..., 0], D[..., 1], D[..., 2] = Dxx, Dyy, Dxy
        v_new = fp2d_sde_step(A, D, v, dt, eps_psd=1e-12)
        dv = v_new[0]
        cov = (dv.T @ dv) / N
        expected = torch.tensor([[Dxx, Dxy], [Dxy, Dyy]], dtype=torch.float64) * dt
        assert torch.allclose(cov, expected, atol=2e-3)


class TestFP2DStep:

    def test_ensure_non_negative_f_clamps_negative_output(self):
        # Divergent A on a positive f produces negative df. The clamp must floor
        # f_new at 0 so the f ≥ 0 invariant holds even under aggressive steps.
        B, Nx, Ny = 1, 6, 6
        f = torch.full((B, Nx, Ny), 0.01)
        Ax = torch.linspace(-10.0, 10.0, Nx).view(Nx, 1).expand(Nx, Ny).contiguous()
        A = torch.stack([Ax, torch.zeros_like(Ax)], dim=0)  # (2, Nx, Ny)
        D = torch.zeros(3, Nx, Ny)
        f_unclamped = fp2d_step(
            A, D, f, dt=1.0, guard_cells=False, ensure_non_negative_f=False
        )
        f_clamped = fp2d_step(
            A, D, f, dt=1.0, guard_cells=False, ensure_non_negative_f=True
        )
        assert (f_unclamped < 0).any()
        assert (f_clamped >= 0).all()

    def test_dt_scalar_and_tensor_consistent(self):
        # Per-batch tensor dt with equal entries must equal scalar dt.
        # Keeps variable-dt batches consistent with the constant-dt path.
        torch.manual_seed(0)
        B, Nx, Ny = 2, 6, 6
        f = torch.rand(B, Nx, Ny)
        A = torch.randn(2, Nx, Ny)
        D = torch.randn(3, Nx, Ny).abs()
        f_scalar = fp2d_step(
            A, D, f, dt=0.05, guard_cells=False, ensure_non_negative_f=False
        )
        f_tensor = fp2d_step(
            A,
            D,
            f,
            dt=torch.full((B,), 0.05),
            guard_cells=False,
            ensure_non_negative_f=False,
        )
        assert torch.allclose(f_scalar, f_tensor)

    def test_guard_cells_preserves_grid_shape(self):
        # The guard-cell branch pads internally then strips back.
        # Output must match the input grid shape regardless of the boundary mode.
        B, Nx, Ny = 2, 6, 6
        f = torch.rand(B, Nx, Ny)
        A = torch.randn(2, Nx, Ny)
        D = torch.randn(3, Nx, Ny).abs()
        for guard in (False, True):
            f_new = fp2d_step(
                A, D, f, dt=0.01, guard_cells=guard, ensure_non_negative_f=False
            )
            assert f_new.shape == f.shape
