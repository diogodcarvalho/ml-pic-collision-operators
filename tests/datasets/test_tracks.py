import yaml
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml_pic_collision_operators.datasets import (
    BaseTracksDataset,
    TemporalUnrolledTracksDataset,
)

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_TRACKS = _EXAMPLES / "dataset_tracks" / "2D" / "normal_-2_0" / "samples"


def _make_tracks_folder(folder, particle_counts, *, i_start=0, i_end=-1, dt=0.5):
    # build a minimal tracks dataset: one h5 DataFrame per dump, sized by particle_counts
    info = {"i_start": i_start, "i_end": i_end, "dt": dt, "v_units": "[c]"}
    with open(folder / "args.yaml", "w") as fh:
        yaml.safe_dump(info, fh)
    for k, n in enumerate(particle_counts):
        df = pd.DataFrame(
            {
                "v1": np.arange(n, dtype=np.float32),
                "v2": np.arange(n, dtype=np.float32),
            },
            index=np.arange(n),
        )
        df.to_hdf(folder / f"{i_start + k:06d}.h5", key="df")
    return folder


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
        assert ds.v_units == "[c]"

    def test_i_end_defaults_to_file_count_when_metadata_unset(self, tmp_path):
        # args.yaml i_end=-1 means use every available dump from i_start onward
        n_files = 4
        _make_tracks_folder(tmp_path, [3] * n_files, i_start=0, i_end=-1)
        ds = BaseTracksDataset(folder=tmp_path)
        assert ds.i_end == ds.i_start + n_files

    def test_coords_inferred_from_file(self):
        ds = BaseTracksDataset(folder=_TRACKS)
        assert ds.coords == ("v1", "v2")

    def test_invalid_mode_raises(self):
        # only 'train' and 'test' are accepted, anything else is a programming error
        with pytest.raises(ValueError):
            BaseTracksDataset(folder=_TRACKS, mode="val")

    def test_load_file_rejects_non_integer_index(self):
        # files are addressed by integer dump index, a non-int request is a bug
        ds = BaseTracksDataset(folder=_TRACKS)
        with pytest.raises(KeyError):
            ds._load_file("0")

    def test_particle_count_mismatch_raises(self, tmp_path):
        # particles are aligned by tag across dumps, a differing count cannot be aligned
        _make_tracks_folder(tmp_path, [5, 4], i_start=0, i_end=-1)
        ds = BaseTracksDataset(folder=tmp_path)
        with pytest.raises(ValueError):
            ds[0]

    def test_targets_are_shifted_inputs(self):
        # per-particle alignment: target at idx must equal input at idx+step_size
        ds = BaseTracksDataset(folder=_TRACKS, step_size=2)
        assert np.allclose(ds[0].targets, ds[2].inputs)
        assert np.allclose(ds[3].targets, ds[5].inputs)

    def test_getitem_dt_scales_with_step_size(self):
        # item dt is the elapsed input->target time: the per-dump dt times step_size,
        # since inputs and targets are step_size dumps apart
        ds1 = BaseTracksDataset(folder=_TRACKS, step_size=1)
        ds2 = BaseTracksDataset(folder=_TRACKS, step_size=2)
        assert ds1[0].dt == pytest.approx(ds1.dt)
        assert ds2[0].dt == pytest.approx(ds2.dt * 2)

    def test_getitem_test_mode_scales_index_by_step_size(self):
        # in test mode idx is multiplied by step_size so pairs are non-overlapping
        step = 2
        ds_test = BaseTracksDataset(folder=_TRACKS, mode="test", step_size=step)
        ds_train = BaseTracksDataset(folder=_TRACKS, mode="train")
        assert np.allclose(ds_test[1].inputs, ds_train[step].inputs)


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

    def test_getitem_dt_scales_with_step_size(self):
        # item dt is the per-step elapsed time: per-dump dt times step_size
        ds1 = TemporalUnrolledTracksDataset(folder=_TRACKS, step_size=1)
        ds2 = TemporalUnrolledTracksDataset(folder=_TRACKS, step_size=2)
        assert ds1[0].dt == pytest.approx(ds1.dt)
        assert ds2[0].dt == pytest.approx(ds2.dt * 2)


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
