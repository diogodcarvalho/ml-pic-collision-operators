from src.models.fokker_planck_2D import FokkerPlanck2D


def test_fokker_planck_2d():
    grid_size = (5, 8)
    grid_dx = (0.2, 0.1)
    model = FokkerPlanck2D(grid_size=grid_size, grid_dx=grid_dx)
    assert model.A.shape == (2, grid_size[0], grid_size[1])
    assert model.B.shape == (2, 2, grid_size[0], grid_size[1])
    print(model.A)
    print(type(model.dx))


test_fokker_planck_2d()
