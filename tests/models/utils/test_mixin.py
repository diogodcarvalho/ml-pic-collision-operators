import pytest
from ml_pic_collision_operators.models import FokkerPlanck2D_Tensor_AD_ParPerp


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
