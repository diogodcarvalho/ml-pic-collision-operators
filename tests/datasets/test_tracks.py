import pytest
import numpy as np
from pathlib import Path

from ml_pic_collision_operators.datasets import (
    BaseTracksDataset,
    TemporalUnrolledTracksDataset,
)

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_TRACKS = _EXAMPLES / "dataset_tracks" / "2D" / "normal_-2_0" / "samples"


class TestBaseTracksDataset:

    def test_kind_and_shape(self):
        ds = BaseTracksDataset(folder=_TRACKS)
        assert ds.kind == "tracks"
        item = ds[0]
        assert item.inputs.shape == (ds.n_particles, 2)
        assert item.targets.shape == (ds.n_particles, 2)

    def test_read_from_samples_args(self):
        ds = BaseTracksDataset(folder=_TRACKS, i_start=0)
        assert ds.dt == pytest.approx(0.831558)
        # args.yaml should overwrite input value
        assert ds.i_start == 5
        assert ds.i_end == 101
        assert ds.grid_units == "[c]"

    def test_coords_inferred_from_file(self):
        ds = BaseTracksDataset(folder=_TRACKS)
        assert ds.coords == ("v1", "v2")

    def test_targets_are_shifted_inputs(self):
        # per-particle alignment: target at idx must equal input at idx+step_size
        ds = BaseTracksDataset(folder=_TRACKS, step_size=2)
        assert np.allclose(ds[0].targets, ds[2].inputs)
        assert np.allclose(ds[3].targets, ds[5].inputs)


class TestTemporalUnrolledTracksDataset:

    def test_targets_have_unroll_leading_dim(self):
        steps = 3
        ds = TemporalUnrolledTracksDataset(folder=_TRACKS, temporal_unroll_steps=steps)
        assert ds[0].targets.shape == (steps, ds.n_particles, 2)

    def test_len_accounts_for_unroll(self):
        steps = 3
        ds = TemporalUnrolledTracksDataset(folder=_TRACKS, temporal_unroll_steps=steps)
        assert len(ds) == ds.i_end - ds.i_start - ds.step_size * steps

    def test_targets_are_shifted_inputs(self):
        # each unroll step must correspond to the input particles at the matching offset
        steps = 2
        ds = TemporalUnrolledTracksDataset(folder=_TRACKS, temporal_unroll_steps=steps)
        ds_base = BaseTracksDataset(folder=_TRACKS, step_size=1)
        item = ds[0]
        for ts in range(steps):
            assert np.allclose(item.targets[ts], ds_base[ts + 1].inputs)


class TestDatasetLengthsHardcoded:
    # Dataset Sizes
    # _TRACKS: i_start=5, i_end=101 (96 frames).

    def test_base_train_step1(self):
        # 96 frames - 1 step = 95 overlapping pairs
        assert len(BaseTracksDataset(folder=_TRACKS, mode="train", step_size=1)) == 95

    def test_base_train_step2(self):
        # 96 frames - 2 steps = 94 overlapping pairs
        assert len(BaseTracksDataset(folder=_TRACKS, mode="train", step_size=2)) == 94

    def test_base_test_step1(self):
        # (96-1)//1 = 95 non-overlapping pairs
        assert len(BaseTracksDataset(folder=_TRACKS, mode="test", step_size=1)) == 95

    def test_base_test_step2(self):
        # (96-1)//2 = 47 non-overlapping pairs; halved relative to step=1
        assert len(BaseTracksDataset(folder=_TRACKS, mode="test", step_size=2)) == 47

    def test_base_train_restricted_i_end(self):
        # i_end=20 → 20-5=15 frames, 15-1=14 pairs
        assert len(BaseTracksDataset(folder=_TRACKS, mode="train", i_end=20)) == 14

    def test_temporal_unroll_steps1(self):
        # 96 - 1*1 = 95, same as base train with step=1
        assert (
            len(TemporalUnrolledTracksDataset(folder=_TRACKS, temporal_unroll_steps=1))
            == 95
        )

    def test_temporal_unroll_steps3(self):
        # 96 - 1*3 = 93; 2 fewer than steps=1 due to 2 extra target slots needed
        assert (
            len(TemporalUnrolledTracksDataset(folder=_TRACKS, temporal_unroll_steps=3))
            == 93
        )

    def test_temporal_unroll_steps3_step_size2(self):
        # 96 - 2*3 = 90
        assert (
            len(
                TemporalUnrolledTracksDataset(
                    folder=_TRACKS, temporal_unroll_steps=3, step_size=2
                )
            )
            == 90
        )
