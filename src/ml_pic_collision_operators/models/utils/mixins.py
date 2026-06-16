from typing import Any


class SupportsAttributeChange:
    """Allows a fixed allowlist of attributes to be mutated after construction.

    Shared across the otherwise-unrelated model base classes so ``change_attribute``
    is defined once instead of copied per base. Subclasses set ``_mutable_attrs`` to
    the attributes they allow to change. The default is empty (nothing is mutable).
    """

    # attributes that may be changed after __init__
    _mutable_attrs: frozenset[str] = frozenset()

    def change_attribute(self, attr_name: str, attr_value: Any) -> None:
        if attr_name in self._mutable_attrs:
            setattr(self, attr_name, attr_value)
        elif hasattr(self, attr_name):
            raise ValueError(
                f"Can not change attribute: {attr_name} after initialization"
            )
        else:
            raise KeyError(f"{type(self)} does not have attribute: {attr_name}")


def grid_shape_checks(
    ndim: int,
    grid_size: tuple[int, ...],
    grid_range: tuple[float, ...],
    grid_dx: tuple[float, ...],
    includes_simmetry: bool,
):
    assert len(grid_size) == ndim
    assert len(grid_range) == 2 * ndim
    assert len(grid_dx) == ndim
    if includes_simmetry:
        for i in range(1, ndim):
            assert grid_size[0] == grid_size[i]
            assert grid_dx[0] == grid_dx[i]
            assert grid_range[0] == grid_range[2 * i]
            assert grid_range[1] == grid_range[2 * i + 1]
        assert grid_range[0] == -grid_range[1]
