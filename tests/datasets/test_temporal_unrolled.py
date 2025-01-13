import numpy as np
from src.datasets import TemporalUnrolledDataset


def test_temporal_unrolled_dataset_dt_1():
    folder = "data/finite_size/0-only-electrons/simulations/linear/data/0/FDIST/2D/uniform_0/f"
    step_size = 2
    data = TemporalUnrolledDataset(
        folder=folder, step_size=step_size, temporal_unroll_steps=1
    )
    assert data.step_size == 2
    assert data.temporal_unroll_steps == 1
    assert len(data) == 104 - step_size
    x, y = next(iter(data))
    x2 = data._load_file(0)
    y2 = data._load_file(2)
    assert np.array_equal(x, x2)
    assert np.array_equal(y[0], y2)
    assert data.grid_ndims == 2
    assert data.grid_size == (50, 50)
    assert np.allclose(data.grid_dx, (0.070 / 50.0, 0.070 / 50.0))


def test_temporal_unrolled_dataset_dt_n():
    folder = "data/finite_size/0-only-electrons/simulations/linear/data/0/FDIST/2D/uniform_0/f"
    step_size = 2
    temporal_unroll_steps = 3
    data = TemporalUnrolledDataset(
        folder=folder, step_size=step_size, temporal_unroll_steps=3
    )
    assert data.step_size == 2
    assert data.temporal_unroll_steps == temporal_unroll_steps
    assert len(data) == 104 - step_size * temporal_unroll_steps
    x, y = next(iter(data))
    x2 = data._load_file(0)
    assert np.array_equal(x, x2)
    for i in range(temporal_unroll_steps):
        y2 = data._load_file(2 * (i + 1))
        assert np.array_equal(y[i], y2)
    assert data.grid_ndims == 2
    assert data.grid_size == (50, 50)
    assert np.allclose(data.grid_dx, (0.070 / 50.0, 0.070 / 50.0))


test_temporal_unrolled_dataset_dt_1()
test_temporal_unrolled_dataset_dt_n()
