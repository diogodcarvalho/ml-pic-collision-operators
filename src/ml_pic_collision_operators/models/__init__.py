from ml_pic_collision_operators.models.fp2d import *
from ml_pic_collision_operators.models.fp3d import *
from ml_pic_collision_operators.models.k2d import *

FPModelType = (
    FokkerPlanck2D_Base
    | FokkerPlanck2D_Base_Conditioned
    | FokkerPlanck2D_Tensor_Base_TimeDependent
    | FokkerPlanck3D_Base
)

KModelType = K2D_Base

ModelType = FPModelType | KModelType
