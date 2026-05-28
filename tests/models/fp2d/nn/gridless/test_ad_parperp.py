import torch

from ml_pic_collision_operators.models.fp2d.nn.gridless.ad_parperp import (
    FokkerPlanck2D_NN_Gridless_AD_ParPerp,
)

from ._ad_shared import GridlessADSharedTests


class TestFokkerPlanck2D_NN_Gridless_AD_ParPerp(GridlessADSharedTests):
    MODEL_CLS = FokkerPlanck2D_NN_Gridless_AD_ParPerp
    HEAD_ATTRS = ("Apar_over_v", "Dperp", "delta_over_v")

    def test_axis_swap_A(self):
        model = self._make_model()
        pts = self._sample_v()
        A = model.A_at_points_real(pts)
        A_swap = model.A_at_points_real(self._swap_axes(pts))
        assert torch.allclose(A_swap[..., 0], A[..., 1], atol=self._ATOL)
        assert torch.allclose(A_swap[..., 1], A[..., 0], atol=self._ATOL)

    def test_axis_swap_D_diagonal(self):
        model = self._make_model()
        pts = self._sample_v()
        D = model.D_at_points_real(pts)
        D_swap = model.D_at_points_real(self._swap_axes(pts))
        assert torch.allclose(D_swap[..., 0], D[..., 1], atol=self._ATOL)
        assert torch.allclose(D_swap[..., 1], D[..., 0], atol=self._ATOL)

    def test_Dxy_invariant_under_axis_swap(self):
        model = self._make_model()
        pts = self._sample_v()
        pts_swap = self._swap_axes(pts)
        D = model.D_at_points_real(pts)
        D_swap = model.D_at_points_real(pts_swap)
        assert torch.allclose(D_swap[..., 2], D[..., 2], atol=self._ATOL)

    def test_origin_boundary_conditions(self):
        # Check A(0) = 0 and D_par(0) = D_perp(0)
        model = self._make_model()
        origin = torch.zeros(1, 3, 2)
        A = model.A_at_points_real(origin)
        D = model.D_at_points_real(origin)
        assert torch.allclose(A, torch.zeros_like(A), atol=self._ATOL)
        assert torch.allclose(D[..., 0], D[..., 1], atol=self._ATOL)
        assert torch.allclose(D[..., 2], torch.zeros_like(D[..., 2]), atol=self._ATOL)
