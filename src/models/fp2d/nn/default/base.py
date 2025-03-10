import torch
import torch.nn as nn

from typing import Callable

from src.models.fp2d.base import FokkerPlanck2DBase
from src.utils import class_from_str


class FokkerPlanck2DNNBase(FokkerPlanck2DBase):
    """Base model to estabilish common structure of Fokker Planck 2D NN classes."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        normalize_v_grid: bool = True,
        includes_symmetry: bool = False,
    ):

        if includes_symmetry:
            assert grid_size[0] == grid_size[1]
            assert grid_range[0] == grid_range[2]
            assert grid_range[1] == grid_range[3]
            assert grid_dx[0] == grid_dx[1]

        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            ensure_non_negative_f=ensure_non_negative_f,
        )

        if isinstance(activation, str):
            activation = class_from_str(activation)

        self._init_NN(
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
        )
        self._init_v_grid(normalize_v_grid)

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        raise NotImplementedError

    def _init_v_grid(self, normalize: bool):
        raise NotImplementedError
