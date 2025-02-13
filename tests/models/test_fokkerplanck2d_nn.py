import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.models import FokkerPlanck2DNN

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def test():
    grid_size = (5, 5)
    grid_dx = (0.1, 0.1)
    grid_range = (-0.01, 0.01, -0.01, 0.01)
    model = FokkerPlanck2DNN(
        grid_size=grid_size,
        grid_dx=grid_dx,
        grid_range=grid_range,
        depth=3,
        width_size=4,
    )
    print(model)

    fig, ax = plt.subplots(1, 2)
    v_grid = model.v_grid.reshape(*grid_size, 2)
    ax[0].imshow(v_grid[..., 0].T, origin="lower")
    ax[1].imshow(v_grid[..., 1].T, origin="lower")
    plt.savefig(os.path.join(FILE_DIR, "test_fokkerplanck2d_nn_vgrid.png"))
    plt.close()
    model.plot(save_to=os.path.join(FILE_DIR, "test_fokkerplanck2d_nn.png"))


test()
