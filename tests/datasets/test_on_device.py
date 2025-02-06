import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.datasets import OnDeviceDataset

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

folder = os.path.join(
    FILE_DIR,
    "../../data/finite_size/0-only-electrons/1-uth-0.01/data/0/FDIST/2D/uniform_-.015_-.005_-.005_.005/range-0.05-nbins-51",
)


def test_on_device_dataset_dt_1():

    step_size = 2
    data = OnDeviceDataset(folder=folder, step_size=step_size, temporal_unroll_steps=1)
    assert data.step_size == step_size
    assert data.temporal_unroll_steps == 1
    assert len(data) == 101 - step_size
    x, y = next(iter(data))
    assert str(x.device) == "cuda:0"
    assert str(y.device) == "cuda:0"
    x2 = data._load_file(0)
    y2 = data._load_file(step_size)
    assert np.allclose(x.cpu().numpy(), x2)
    assert np.allclose(y[0].cpu().numpy(), y2)
    assert data.grid_ndims == 2
    assert data.grid_size == (51, 51)
    assert np.allclose(data.grid_dx, (0.1 / 51, 0.1 / 51))

    n_steps = 3
    fig, ax = plt.subplots(n_steps, 2, figsize=(3, n_steps / 2 * 3))
    dataloader = iter(data)
    for i in range(n_steps):
        x, y = next(dataloader)
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
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
    fig.savefig(
        os.path.join(FILE_DIR, f"test_on_device-unroll-1-step_size-{step_size}.png")
    )


def test_on_device_dataset_dt_n():
    for step_size in [1, 2]:
        temporal_unroll_steps = 4
        data = OnDeviceDataset(
            folder=folder,
            step_size=step_size,
            temporal_unroll_steps=temporal_unroll_steps,
        )
        assert data.step_size == step_size
        assert data.temporal_unroll_steps == temporal_unroll_steps
        assert len(data) == 101 - step_size * temporal_unroll_steps
        x, y = next(iter(data))
        assert str(x.device) == "cuda:0"
        assert str(y.device) == "cuda:0"
        x2 = data._load_file(0)
        assert np.allclose(x.cpu().numpy(), x2)
        for i in range(temporal_unroll_steps):
            y2 = data._load_file(step_size * (i + 1))
            assert np.allclose(y[i].cpu().numpy(), y2)
        assert data.grid_ndims == 2
        assert data.grid_size == (51, 51)
        assert np.allclose(data.grid_dx, (0.1 / 51, 0.1 / 51))

        n_steps = 3
        fig, ax = plt.subplots(
            n_steps,
            temporal_unroll_steps + 1,
            figsize=(3 * (1 + temporal_unroll_steps), n_steps * 3),
        )
        dataloader = iter(data)
        for i in range(n_steps):
            x, y = next(dataloader)
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            vmax = np.max(np.abs(x))
            kwargs = {"cmap": "Reds", "vmax": vmax, "vmin": 0, "origin": "lower"}
            ax[i, 0].imshow(x.T, **kwargs)
            for j in range(temporal_unroll_steps):
                ax[i, 1 + j].imshow(y[j].T, **kwargs)
                ax[i, 1 + j].set_title(rf"Target_{j} ($\Delta i={step_size*(j+1)}$)")
            ax[i, 0].set_ylabel(f"i={i}")
        ax[0, 0].set_title("Input")
        for a in ax.flatten():
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                FILE_DIR,
                f"test_on_device-unroll-{temporal_unroll_steps}-step_size-{step_size}.png",
            )
        )


test_on_device_dataset_dt_1()
test_on_device_dataset_dt_n()
