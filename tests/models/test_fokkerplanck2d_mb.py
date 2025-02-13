import os
import jax.numpy as jnp
from src.models import FokkerPlanck2Dmb

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def test():
    grid_size = (11, 11)
    grid_dx = (0.1, 0.1)
    grid_range = (-0.01, 0.01, -0.01, 0.01)
    model = FokkerPlanck2Dmb(
        grid_size=grid_size, grid_dx=grid_dx, grid_range=grid_range
    )
    print(model.m)
    assert model.m.shape == (1,)
    assert model.b.shape == (1,)
    assert model.b == 1
    assert jnp.array_equal(
        model.B_grid,
        jnp.stack(
            [jnp.ones(grid_size), jnp.ones(grid_size), jnp.zeros(grid_size)], axis=0
        ),
    )
    model.plot(save_to=os.path.join(FILE_DIR, "test_fokkerplanck2d_mb.png"))


test()
