import pytest
import torch

from ml_pic_collision_operators.losses import (
    ConcatTestFunctions,
    GaussianTestFunctions,
    MonomialTestFunctions,
    RadialQuadraticGaussianTestFunctions,
)
from ml_pic_collision_operators.losses.test_functions.base import TestFunction

_ATOL = 1e-8
_TEST_FUNCTIONS = [
    MonomialTestFunctions(n_dims=2, degree=3),
    MonomialTestFunctions(n_dims=3, degree=2),
    GaussianTestFunctions(centers=[[0.0, 0.0], [0.3, -0.2]], sigma=0.5),
    GaussianTestFunctions(centers=[[0.0, 0.0, 0.0]], sigma=[0.4]),
    RadialQuadraticGaussianTestFunctions(centers=[[0.0, 0.0], [0.2, -0.1]], sigma=0.4),
    RadialQuadraticGaussianTestFunctions(centers=[[0.0, 0.0, 0.0]], sigma=[0.3]),
    ConcatTestFunctions(
        [
            MonomialTestFunctions(n_dims=2, degree=2),
            GaussianTestFunctions(centers=[[0.0, 0.0]], sigma=0.3),
            RadialQuadraticGaussianTestFunctions(centers=[[0.1, 0.1]], sigma=0.5),
        ]
    ),
]


class TestBaseTestFunction:

    def test_evaluate_phi(self):
        # ensure that test functions that do not overwrite evaluate_phi only return phi
        class _DummyTf(TestFunction):
            n_dims = 2
            n_functions = 1

            def evaluate(self, x):
                phi = x.sum(dim=-1, keepdim=True)
                return phi, None, None

        tf = _DummyTf()
        x = torch.randn(2, 4, 2)
        phi, _, _ = tf.evaluate(x)
        assert torch.equal(tf.evaluate_phi(x), phi)


class TestAllTestFunctions:

    @pytest.mark.parametrize("tf", _TEST_FUNCTIONS)
    def test_derivatives_match_autograd(self, tf):
        B, N, D = 2, 4, tf.n_dims
        x = torch.randn(B, N, D, requires_grad=True)
        phi, grad, hess = tf.evaluate(x)
        assert phi.shape == (B, N, tf.n_functions)
        assert grad.shape == (B, N, tf.n_functions, D)
        assert hess.shape == (B, N, tf.n_functions, D, D)
        for k in range(tf.n_functions):
            g_auto = torch.autograd.grad(
                phi[..., k].sum(), x, create_graph=True, retain_graph=True
            )[0]
            assert torch.allclose(g_auto, grad[..., k, :], atol=_ATOL)
            for j in range(D):
                h_auto = torch.autograd.grad(
                    grad[..., k, j].sum(), x, retain_graph=True
                )[0]
                assert torch.allclose(h_auto, hess[..., k, j, :], atol=_ATOL)

    @pytest.mark.parametrize("tf", _TEST_FUNCTIONS)
    def test_evaluate_phi_matches_evaluate(self, tf):
        x = torch.randn(2, 4, tf.n_dims)
        phi_full, _, _ = tf.evaluate(x)
        phi_only = tf.evaluate_phi(x)
        assert phi_only.shape == phi_full.shape
        assert torch.equal(phi_only, phi_full)

    @pytest.mark.parametrize("tf", _TEST_FUNCTIONS)
    def test_evaluate_rejects_wrong_last_dim(self, tf):
        x = torch.randn(1, 3, tf.n_dims + 1)
        with pytest.raises(ValueError, match="n_dims"):
            tf.evaluate(x)
        with pytest.raises(ValueError, match="n_dims"):
            tf.evaluate_phi(x)


class TestMonomialConstruction:

    def test_rejects_zero_n_dims(self):
        with pytest.raises(ValueError, match="n_dims"):
            MonomialTestFunctions(n_dims=0, degree=2)

    def test_rejects_zero_degree(self):
        with pytest.raises(ValueError, match="degree"):
            MonomialTestFunctions(n_dims=2, degree=0)

    def test_n_functions_matches_multi_index_count(self):
        # K = C(D + degree, D) - 1 (all multi-indices of total degree 1..degree)
        # D=2, degree=3 -> C(5,2) - 1 = 9
        assert MonomialTestFunctions(n_dims=2, degree=3).n_functions == 9
        # D=3, degree=2 -> C(5,3) - 1 = 9
        assert MonomialTestFunctions(n_dims=3, degree=2).n_functions == 9


class TestGaussianConstruction:

    def test_rejects_1d_centers(self):
        with pytest.raises(ValueError, match="centers"):
            GaussianTestFunctions(centers=[0.0, 0.0], sigma=0.5)

    def test_rejects_wrong_sigma_shape(self):
        with pytest.raises(ValueError, match="sigma"):
            GaussianTestFunctions(centers=[[0.0, 0.0], [0.1, 0.1]], sigma=[0.5])

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            GaussianTestFunctions(centers=[[0.0, 0.0]], sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            GaussianTestFunctions(centers=[[0.0, 0.0]], sigma=[-0.1])

    def test_scalar_sigma_broadcasts(self):
        # Scalar sigma must produce the same phi as the explicitly broadcast vector.
        centers = [[0.0, 0.0], [0.1, 0.1]]
        scalar = GaussianTestFunctions(centers=centers, sigma=0.3)
        vector = GaussianTestFunctions(centers=centers, sigma=[0.3, 0.3])
        x = torch.randn(1, 4, 2)
        assert torch.equal(scalar.evaluate_phi(x), vector.evaluate_phi(x))


class TestRadialQuadraticGaussianConstruction:

    def test_rejects_1d_centers(self):
        with pytest.raises(ValueError, match="centers"):
            RadialQuadraticGaussianTestFunctions(centers=[0.0, 0.0], sigma=0.5)

    def test_rejects_wrong_sigma_shape(self):
        with pytest.raises(ValueError, match="sigma"):
            RadialQuadraticGaussianTestFunctions(
                centers=[[0.0, 0.0], [0.1, 0.1]], sigma=[0.5]
            )

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            RadialQuadraticGaussianTestFunctions(centers=[[0.0, 0.0]], sigma=0.0)


class TestConcatTestFunctions:

    def test_stacks_along_k(self):
        a = MonomialTestFunctions(n_dims=2, degree=2)
        b = GaussianTestFunctions(centers=[[0.0, 0.0]], sigma=0.3)
        composite = ConcatTestFunctions([a, b])
        assert composite.n_functions == a.n_functions + b.n_functions
        x = torch.randn(1, 5, 2)
        phi, grad, hess = composite.evaluate(x)
        assert phi.shape[-1] == composite.n_functions
        assert grad.shape[-2] == composite.n_functions
        assert hess.shape[-3] == composite.n_functions

    def test_rejects_dim_mismatch(self):
        a = MonomialTestFunctions(n_dims=2, degree=2)
        b = MonomialTestFunctions(n_dims=3, degree=2)
        with pytest.raises(ValueError, match="n_dims"):
            ConcatTestFunctions([a, b])

    def test_rejects_empty_components(self):
        with pytest.raises(ValueError, match="at least one"):
            ConcatTestFunctions([])

    def test_phi_matches_concatenation_of_children(self):
        # Sanity: evaluate_phi on the composite equals torch.cat of children's phi.
        a = MonomialTestFunctions(n_dims=2, degree=2)
        b = GaussianTestFunctions(centers=[[0.0, 0.0]], sigma=0.3)
        composite = ConcatTestFunctions([a, b])
        x = torch.randn(1, 5, 2)
        expected = torch.cat([a.evaluate_phi(x), b.evaluate_phi(x)], dim=-1)
        assert torch.equal(composite.evaluate_phi(x), expected)
