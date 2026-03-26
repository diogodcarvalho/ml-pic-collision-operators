from .base import FokkerPlanck2D_Base
from .base_conditioned import FokkerPlanck2DBaseConditioned
from .tensor import *
from .nn import *

FPModelType = FokkerPlanck2D_Base | FokkerPlanck2DBaseConditioned
