import pytest
from ml_pic_collision_operators.models import FokkerPlanck2D_Tensor_AD_ParPerp
from ml_pic_collision_operators.models.utils import grid_shape_checks


def _model():
    # Picked a model that inherits change_attribute from the SupportsAttributeChange
    return FokkerPlanck2D_Tensor_AD_ParPerp(
        grid_size=(5, 5),
        grid_range=(-1.0, 1.0, -1.0, 1.0),
        grid_dx=(0.4, 0.4),
        grid_units="[c]",
    )


class TestChangeAttribute:

    def test_mutable_attribute_is_updated(self):
        model = _model()
        model.change_attribute("guard_cells", True)
        assert model.guard_cells is True

    def test_immutable_attribute_raises_value_error(self):
        with pytest.raises(ValueError, match="Can not change attribute"):
            _model().change_attribute("grid_units", "[v_th]")

    def test_unknown_attribute_raises_key_error(self):
        with pytest.raises(KeyError, match="does not have attribute"):
            _model().change_attribute("does_not_exist", 0)


_VALID_GRIDS = {
    "1D": dict(
        ndim=1,
        grid_size=(4,),
        grid_range=(-1.0, 1.0),
        grid_dx=(0.5,),
    ),
    "2D": dict(
        ndim=2,
        grid_size=(4, 4),
        grid_range=(-1.0, 1.0, -1.0, 1.0),
        grid_dx=(0.5, 0.5),
    ),
    "3D": dict(
        ndim=3,
        grid_size=(4, 4, 4),
        grid_range=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        grid_dx=(0.5, 0.5, 0.5),
    ),
}


class TestGridShapeChecks:

    @pytest.mark.parametrize("grid", _VALID_GRIDS.values(), ids=_VALID_GRIDS.keys())
    def test_valid_symmetric_grid_passes(self, grid):
        # a consistent, zero-centered symmetric grid must not raise
        grid_shape_checks(**grid, includes_simmetry=True)

    @pytest.mark.parametrize(
        "base, overrides, includes_simmetry",
        [
            # wrong tuple lengths (symmetry irrelevant)
            ("2D", {"grid_size": (4, 4, 4)}, False),
            ("2D", {"grid_range": (-1.0, 1.0, -1.0)}, False),
            ("2D", {"grid_dx": (0.5,)}, False),
            # symmetry violations
            ("2D", {"grid_size": (4, 5)}, True),
            ("3D", {"grid_size": (4, 4, 5)}, True),
            ("2D", {"grid_dx": (0.5, 0.4)}, True),
            ("3D", {"grid_dx": (0.5, 0.5, 0.4)}, True),
            ("2D", {"grid_range": (-1.0, 1.0, -1.0, 2.0)}, True),
            ("3D", {"grid_range": (-1.0, 1.0, -1.0, 1.0, -1.0, 2.0)}, True),
            ("1D", {"grid_range": (0.0, 2.0)}, True),
            ("2D", {"grid_range": (0.0, 2.0, 0.0, 2.0)}, True),
            ("3D", {"grid_range": (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)}, True),
        ],
        ids=[
            "grid_size_2d",
            "grid_range_2d",
            "grid_dx_2d",
            "sym_grid_size_2d",
            "sym_grid_size_3d",
            "sym_grid_dx_2d",
            "sym_grid_dx_3d",
            "sym_grid_range_2d",
            "sym_grid_range_3d",
            "sym_grid_range_not_centered_1d",
            "sym_grid_range_not_centered_2d",
            "sym_grid_range_not_centered_3d",
        ],
    )
    def test_invalid_grid_raises(self, base, overrides, includes_simmetry):
        # start from a passing grid and break only the field under test
        grid = {**_VALID_GRIDS[base], **overrides}
        with pytest.raises(AssertionError):
            grid_shape_checks(**grid, includes_simmetry=includes_simmetry)
