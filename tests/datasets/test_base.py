import numpy as np
from src.datasets.base import BaseDataset


def test_basedataset():
    folder = (
        "data/finite_size/0-only-electrons/simulations/linear/data/0/FDIST/2D/uniform_0"
    )
    step_size = 2
    data = BaseDataset(folder=folder, step_size=step_size)
    assert data.step_size == 2
    assert len(data) == 104 - step_size
    x, y = next(iter(data))
    x2 = data._load_file(0)
    y2 = data._load_file(2)
    assert np.array_equal(x, x2)
    assert np.array_equal(y, y2)
    assert data.grid_ndims == 2
    assert data.grid_size == (50, 50)
    assert np.allclose(data.grid_dx, (0.070 / 50.0, 0.070 / 50.0))


test_basedataset()
