import os
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.base import BaseDataset
from src.dataloaders.base import BaseDataLoader

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basedataloader():
    folder = os.path.join(
        FILE_DIR,
        "../../data/finite_size/0-only-electrons/1-uth-0.01/data/0/FDIST/2D/uniform_-.015_-.005_-.005_.005/range-0.05-nbins-51",
    )

    for step_size in [1, 2, 3]:
        data = BaseDataset(folder=folder, step_size=step_size)

        dataloader = BaseDataLoader(data)
        x, y = next(iter(dataloader))
        x2, y2 = data[0]
        print(type(x), type(y), x.shape, y.shape)
        print(type(x2), type(y2), x2.shape, y2.shape)
        assert np.allclose(np.asarray(x[0]), x2, atol=1e-12)
        assert np.allclose(np.asarray(y[0]), y2, atol=1e-12)

        n_steps = 4
        fig, ax = plt.subplots(n_steps, 2, figsize=(3, n_steps / 2 * 3))
        dataloader = iter(dataloader)
        for i in range(n_steps):
            x, y = next(dataloader)
            x = np.array(x[0])
            y = np.array(y[0])
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


test_basedataloader()
