import numpy as np
from src.datasets.base import BaseDataset
from src.dataloaders.base import BaseDataLoader


def test_basedataloader():
    folder = "data/finite_size/0-only-electrons/simulations/linear/data/0/FDIST/2D/uniform_0/f"
    step_size = 2
    data = BaseDataset(folder=folder, step_size=step_size)

    dataloader = BaseDataLoader(data)
    x, y = next(iter(dataloader))
    x2, y2 = data[0]
    assert np.array_equal(x[0], x2)
    assert np.array_equal(y[0], y2)


test_basedataloader()
