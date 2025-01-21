from src.models.fokker_planck_2D import FokkerPlanck2D


def test_fokker_planck_2d():
    grid_size = (5, 8)
    grid_dx = (0.2, 0.1)
    grid_range = (-0.01, 0.01, -0.012, 0.012)
    model = FokkerPlanck2D(grid_size=grid_size, grid_dx=grid_dx, grid_range=grid_range)
    assert model.A.shape == (2, grid_size[0], grid_size[1])
    assert model.B.shape == (3, grid_size[0], grid_size[1])
    model.plot(save_to="test_fp2d.png")


test_fokker_planck_2d()
