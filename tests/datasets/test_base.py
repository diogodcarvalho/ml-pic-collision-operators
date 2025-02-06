import os
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.base import BaseDataset

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basedataset():
    folder = os.path.join(
        FILE_DIR,
        "../../data/finite_size/0-only-electrons/1-uth-0.01/data/0/FDIST/2D/uniform_-.015_-.005_-.005_.005/range-0.05-nbins-51",
    )

    for step_size in [1, 2, 3]:

        data = BaseDataset(folder=folder, step_size=step_size)
        assert data.step_size == step_size
        assert len(data) == 101 - step_size
        x, y = next(iter(data))
        x2 = data._load_file(0)
        y2 = data._load_file(step_size)
        assert np.array_equal(x, x2)
        assert np.array_equal(y, y2)
        assert data.grid_ndims == 2
        assert data.grid_size == (51, 51)
        assert np.allclose(data.grid_dx, (0.1 / 51, 0.1 / 51))

        n_steps = 3
        fig, ax = plt.subplots(n_steps, 2, figsize=(3, n_steps / 2 * 3))
        dataloader = iter(data)
        for i in range(n_steps):
            x, y = next(dataloader)
            x = np.array(x)
            y = np.array(y)
            vmax = np.max(np.abs(x))
            kwargs = {"cmap": "Reds", "vmax": vmax, "vmin": 0, "origin": "lower"}
            ax[i, 0].imshow(x.T, **kwargs)
            ax[i, 1].imshow(y.T, **kwargs)
            ax[i, 0].set_ylabel(f"i={i}")
        ax[0, 0].set_title("Input")
        ax[0, 1].set_title(rf"Target ($\Delta i={step_size}$)")
        for a in ax.flatten():
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(FILE_DIR, f"test_base-step_size-{step_size}.png"))


test_basedataset()
