import pytest
import numpy as np
import pandas as pd
import h5py  # type: ignore[import-untyped]

from ml_pic_collision_operators.logging import (
    load_model_from_AB_hdf,
)


def test_load_model_from_AB_hdf(tmp_path):

    file_path = tmp_path / "test_data.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("grid_size", data=[4, 4])
        f.create_dataset("grid_dx", data=[0.5, 0.5])
        f.create_dataset("grid_range", data=[-1.0, 1.0, -1.0, 1.0])
        # Using units of [c] to test conversion to [v_th]
        f.create_dataset("grid_range_units", data="[c]")
        # v_th != 1.0 to test unit conversion
        f.create_dataset("v_th", data=2.0)
        # A, B all ones for simplicity
        f.create_dataset("A", data=np.ones((2, 4, 4)))
        f.create_dataset("B", data=np.ones((3, 4, 4)))

    model = load_model_from_AB_hdf(str(file_path), includes_time=False)
    assert model.grid_range == [-0.5, 0.5, -0.5, 0.5]
    A_result = model.A.detach().numpy()
    B_result = model.B.detach().numpy()
    assert np.allclose(
        A_result, 2.0
    ), f"A normalization failed: expected 2.0, got {A_result.max()}"
    assert np.allclose(
        B_result, 4.0
    ), f"B normalization failed: expected 4.0, got {B_result.max()}"


def test_load_AB_model_invalid_units(tmp_path):
    file_path = tmp_path / "bad_units.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("grid_range_units", data=b"[meters]")
        # ... (add other required keys with dummy data)
        f.create_dataset("grid_size", data=[1, 1])
        f.create_dataset("grid_dx", data=[1, 1])
        f.create_dataset("grid_range", data=[0, 1, 0, 1])
        f.create_dataset("v_th", data=1.0)
        f.create_dataset("A", data=np.zeros((2, 1, 1)))
        f.create_dataset("B", data=np.zeros((3, 1, 1)))

    with pytest.raises(Exception, match="non-accepted units"):
        load_model_from_AB_hdf(str(file_path))
