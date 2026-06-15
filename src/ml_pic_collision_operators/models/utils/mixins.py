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
