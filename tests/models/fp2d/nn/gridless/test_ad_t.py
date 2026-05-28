import torch

from ml_pic_collision_operators.models.fp2d.nn.gridless.ad_t import (
    FokkerPlanck2D_NN_Gridless_AD_T,
)

from ._ad_shared import GridlessADSharedTests


class TestFokkerPlanck2D_NN_Gridless_AD_T(GridlessADSharedTests):
    MODEL_CLS = FokkerPlanck2D_NN_Gridless_AD_T
    HEAD_ATTRS = ("Ax", "Dxx", "Dxy")

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
