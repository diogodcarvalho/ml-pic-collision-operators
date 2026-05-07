import numpy as np
import pytest
from pathlib import Path

from ml_pic_collision_operators.datasets import (
    BaseDataset,
    BasewConditionersDataset,
    DatasetItem,
    TemporalUnrolledDataset,
    TemporalUnrolledwConditionersDataset,
)

_EXAMPLES = Path(__file__).parents[1] / "examples" / "dataset"
_DS_2D = _EXAMPLES / "normal_-2_0" / "f"
_DS_3D = _EXAMPLES / "normal_-2_0_0_3D" / "f"


class TestBaseDataset:

    def test_loads_metadata_from_args_yaml(self):
        # ensures args.yaml is parsed and drives grid/physics properties
        ds = BaseDataset(_DS_2D)
        assert ds.dt == pytest.approx(0.9978696)
        assert ds.grid_ndims == 2
        assert ds.grid_size == (51, 51)
        assert ds.original_grid_range == [-5.0, 5.0, -5.0, 5.0]

    def test_len_train_mode(self):
        # train mode yields all overlapping (input, target) pairs
        ds = BaseDataset(_DS_2D, mode="train")
        assert len(ds) == (ds.i_end - ds.i_start) - ds.step_size

    def test_len_test_mode_nonoverlapping(self):
        # test mode yields non-overlapping sequential pairs, fewer than train for step_size>1
        ds = BaseDataset(_DS_2D, mode="test", step_size=2)
        n = ds.i_end - ds.i_start
        assert len(ds) == (n - 1) // 2

    def test_getitem_shapes_match_grid(self):
        # inputs and targets must have the same shape as the distribution grid
        ds = BaseDataset(_DS_2D)
        item = ds[0]
        assert isinstance(item, DatasetItem)
        assert item.inputs.shape == ds.grid_size
        assert item.targets.shape == ds.grid_size

    def test_normalization_sums_to_one(self):
        # each frame divided by n_particles so it integrates to 1 over velocity space
        ds = BaseDataset(_DS_2D)
        item = ds[0]
        assert item.inputs.sum() == pytest.approx(1.0, rel=1e-5)
        assert item.targets.sum() == pytest.approx(1.0, rel=1e-5)

    def test_extra_cells_expands_grid_size(self):
        # extra_cells pads both sides in every dimension, growing grid by 2*extra_cells per axis
        extra = 3
        ds = BaseDataset(_DS_2D, extra_cells=extra)
        expected = tuple(s + 2 * extra for s in ds.original_grid_size)
        assert ds.grid_size == expected
        assert ds[0].inputs.shape == expected

    def test_invalid_mode_raises(self):
        # only 'train' and 'test' are valid; anything else is a programming error
        with pytest.raises(ValueError):
            BaseDataset(_DS_2D, mode="val")

    def test_i_start_uses_metadata_floor(self):
        # passing i_start below the metadata minimum must be silently clipped
        ds = BaseDataset(_DS_2D, i_start=0)
        assert ds.i_start == 5  # args.yaml sets i_start=5 as minimum

    def test_i_end_limits_available_samples(self):
        # restricting i_end must yield a strictly shorter dataset
        ds_full = BaseDataset(_DS_2D)
        ds_limited = BaseDataset(_DS_2D, i_end=ds_full.i_start + 10)
        assert len(ds_limited) < len(ds_full)

    def test_step_size_offsets_target_by_step(self):
        # target from step_size=2 at idx=0 must equal input from step_size=1 at idx=2
        ds2 = BaseDataset(_DS_2D, step_size=2)
        ds1 = BaseDataset(_DS_2D, step_size=1)
        assert np.allclose(ds2[0].targets, ds1[2].inputs)

    def test_3d_dataset_grid_ndims(self):
        # 3D velocity grids must produce ndims=3 and a length-3 grid_size
        ds = BaseDataset(_DS_3D)
        assert ds.grid_ndims == 3
        assert len(ds.grid_size) == 3

    def test_getitem_dt_matches_metadata(self):
        # dt in each item must be the simulation timestep, not a computed value
        ds = BaseDataset(_DS_2D)
        assert ds[0].dt == pytest.approx(ds.dt)


class TestBasewConditionersDataset:

    def test_no_conditioners_returns_empty_array(self):
        # None conditioners must produce an empty array so downstream code can always concatenate
        ds = BasewConditionersDataset(_DS_2D)
        item = ds[0]
        assert item.conditioners is not None
        assert item.conditioners.shape == (0,)

    def test_conditioners_array_preserves_order(self):
        # conditioner values must appear in dict-insertion order
        cond = {"ppc": 300.0, "v_th": 0.05, "shape": 2.0}
        ds = BasewConditionersDataset(_DS_2D, conditioners=cond)
        item = ds[0]
        assert item.conditioners.shape == (3,)
        assert np.allclose(item.conditioners, [300.0, 0.05, 2.0])

    def test_conditioners_size_includes_time_flag(self):
        # conditioners_size must count the time entry when include_time=True
        cond = {"a": 1.0, "b": 2.0}
        ds_no_time = BasewConditionersDataset(
            _DS_2D, conditioners=cond, include_time=False
        )
        ds_time = BasewConditionersDataset(_DS_2D, conditioners=cond, include_time=True)
        assert ds_no_time.conditioners_size == 2
        assert ds_time.conditioners_size == 3

    def test_include_time_appends_after_conditioners(self):
        # time is concatenated *after* other conditioners (at[-1]), not before
        cond = {"a": 1.0}
        ds = BasewConditionersDataset(_DS_2D, conditioners=cond, include_time=True)
        item0 = ds[0]
        assert item0.conditioners.shape == (2,)
        assert item0.conditioners[0] == pytest.approx(1.0)  # conditioner first
        assert item0.conditioners[-1] == pytest.approx(0.0)  # time=dt*0 at idx=0

    def test_include_time_scales_with_index(self):
        # time value must increment by dt for each successive item
        ds = BasewConditionersDataset(_DS_2D, include_time=True)
        assert ds[0].conditioners[-1] == pytest.approx(0.0)
        assert ds[1].conditioners[-1] == pytest.approx(ds.dt)


class TestTemporalUnrolledDataset:

    def test_len_accounts_for_unroll_steps(self):
        # length must shrink to leave room for all unroll targets at the end of the range
        steps = 3
        ds = TemporalUnrolledDataset(_DS_2D, temporal_unroll_steps=steps)
        assert len(ds) == ds.i_end - ds.i_start - ds.step_size * steps

    def test_targets_have_unroll_leading_dim(self):
        # targets must be stacked along a new axis of size temporal_unroll_steps
        steps = 3
        ds = TemporalUnrolledDataset(_DS_2D, temporal_unroll_steps=steps)
        assert ds[0].targets.shape == (steps, *ds.grid_size)

    def test_targets_are_consecutive_files(self):
        # each unroll step must correspond to the same frame a BaseDataset would load at that offset
        steps = 2
        ds = TemporalUnrolledDataset(_DS_2D, temporal_unroll_steps=steps)
        ds_base = BaseDataset(_DS_2D, step_size=1)
        item = ds[0]
        for ts in range(steps):
            assert np.allclose(item.targets[ts], ds_base[ts + 1].inputs)


class TestDatasetLengthsHardcoded:
    # Dataset Sizes
    # _DS_2D: i_start=5, i_end=96 (91 frames).
    # _DS_3D: i_start=5, i_end=196 (191 frames).

    def test_base_train_step1(self):
        # 91 frames - 1 step = 90 overlapping pairs
        assert len(BaseDataset(_DS_2D, mode="train", step_size=1)) == 90

    def test_base_train_step2(self):
        # 91 frames - 2 steps = 89 overlapping pairs
        assert len(BaseDataset(_DS_2D, mode="train", step_size=2)) == 89

    def test_base_test_step1(self):
        # (91-1)//1 = 90 non-overlapping pairs
        assert len(BaseDataset(_DS_2D, mode="test", step_size=1)) == 90

    def test_base_test_step2(self):
        # (91-1)//2 = 45 non-overlapping pairs; halved relative to step=1
        assert len(BaseDataset(_DS_2D, mode="test", step_size=2)) == 45

    def test_base_train_restricted_i_end(self):
        # i_end=20 → 20-5=15 frames, 15-1=14 pairs
        assert len(BaseDataset(_DS_2D, mode="train", i_end=20)) == 14

    def test_temporal_unroll_steps1(self):
        # 91 - 1*1 = 90, same as base train with step=1
        assert len(TemporalUnrolledDataset(_DS_2D, temporal_unroll_steps=1)) == 90

    def test_temporal_unroll_steps3(self):
        # 91 - 1*3 = 88; 2 fewer than steps=1 due to 2 extra target slots needed
        assert len(TemporalUnrolledDataset(_DS_2D, temporal_unroll_steps=3)) == 88

    def test_temporal_unroll_steps3_step_size2(self):
        # 91 - 2*3 = 85
        assert (
            len(TemporalUnrolledDataset(_DS_2D, temporal_unroll_steps=3, step_size=2))
            == 85
        )

    def test_base_3d_train_step1(self):
        # 191 frames - 1 = 190 pairs
        assert len(BaseDataset(_DS_3D, mode="train", step_size=1)) == 190


class TestTemporalUnrolledwConditionersDataset:

    def test_time_appended_after_conditioners(self):
        # time must be last ([-1]), consistent with BasewConditionersDataset and test.py:343
        cond = {"a": 1.0}
        ds = TemporalUnrolledwConditionersDataset(
            _DS_2D, conditioners=cond, include_time=True
        )
        item0 = ds[0]
        assert item0.conditioners.shape == (2,)
        assert item0.conditioners[0] == pytest.approx(1.0)   # conditioner first
        assert item0.conditioners[-1] == pytest.approx(0.0)  # time last

    def test_conditioners_do_not_affect_targets_shape(self):
        # temporal unrolling of targets must be independent of conditioners
        steps = 2
        ds = TemporalUnrolledwConditionersDataset(
            _DS_2D, conditioners={"ppc": 300.0}, temporal_unroll_steps=steps
        )
        item = ds[0]
        assert item.targets.shape == (steps, *ds.grid_size)
        assert item.conditioners.shape == (1,)
