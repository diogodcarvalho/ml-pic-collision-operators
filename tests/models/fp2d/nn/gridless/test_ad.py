from ml_pic_collision_operators.models.fp2d.nn.gridless.ad import (
    FokkerPlanck2D_NN_Gridless_AD,
)

from ._ad_shared import GridlessADSharedTests


class TestFokkerPlanck2D_NN_Gridless_AD(GridlessADSharedTests):
    MODEL_CLS = FokkerPlanck2D_NN_Gridless_AD
    HEAD_ATTRS = ("Ax", "Ay", "Dxx", "Dyy", "Dxy")
