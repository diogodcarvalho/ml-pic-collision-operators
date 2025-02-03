from src.models.fokker_planck_2D_minimal import FokkerPlanck2DMinimal


def test_fokker_planck_2d():
    grid_size = (5, 5)
    grid_dx = (0.1, 0.1)
    grid_range = (-0.01, 0.01, -0.01, 0.01)
    model = FokkerPlanck2DMinimal(
        grid_size=grid_size, grid_dx=grid_dx, grid_range=grid_range
    )
    print(model.A_r)
    assert model.A_r.shape == (grid_size[0] // 2 + grid_size[0] % 2, 1)
    assert model.b.shape == (1,)
    model.plot(save_to="test_fp2d.png")


test_fokker_planck_2d()
