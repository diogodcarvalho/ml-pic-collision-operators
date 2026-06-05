import math
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
import torch

# No display needed
matplotlib.use("Agg")

from ml_pic_collision_operators.config.train import TestFunctionConfig
from ml_pic_collision_operators.losses.test_functions import (
    ConcatTestFunctions,
    MonomialTestFunctions,
)
from ml_pic_collision_operators.test import (
    _build_test_functions,
    _histogram_from_tracks,
    _compute_all_metrics,
    _generate_video_from_frames,
    plot_hist_comparison,
    plot_scatter_comparison,
)


class TestPlotErrorBranches:
    def test_hist_requires_plot_slice_for_3d(self):
        f = np.zeros((1, 2, 2, 2))  # ndim == 4 means a 3D fdist with a batch axis
        with pytest.raises(ValueError, match="plot_slice"):
            plot_hist_comparison(
                f, f, bin_range=[0, 1] * 3, bin_units="", plot_slice=None
            )

    def test_scatter_rejects_non_2d(self):
        v = np.zeros((10, 3))  # 3D phase space
        with pytest.raises(NotImplementedError):
            plot_scatter_comparison(v, v, bin_range=(0, 1, 0, 1, 0, 1), bin_units="")


class TestComputeAllMetrics:

    Y_TRUE = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    Y_PRED = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

    def test_each_metric_value(self):
        all_metrics = ["mse", "l1", "l2", "l1_norm", "l2_norm"]
        values = _compute_all_metrics(self.Y_TRUE, self.Y_PRED, all_metrics)

        assert values["mse"] == pytest.approx(0.25)
        assert values["l1"] == pytest.approx(1.0)
        assert values["l2"] == pytest.approx(1.0)
        # l1_norm divides by sum(y_true), l2_norm divides by the L2 norm of y_true.
        assert values["l1_norm"] == pytest.approx(1.0 / 10.0)
        assert values["l2_norm"] == pytest.approx(1.0 / math.sqrt(30.0))

    def test_norm_metrics_without_their_base_metric(self):
        # Requesting *_norm without l1/l2 must recompute them internally.
        values = _compute_all_metrics(self.Y_TRUE, self.Y_PRED, ["l1_norm", "l2_norm"])
        assert values["l1_norm"] == pytest.approx(1.0 / 10.0)
        assert values["l2_norm"] == pytest.approx(1.0 / math.sqrt(30.0))

    def test_only_requested_metrics_returned(self):
        values = _compute_all_metrics(self.Y_TRUE, self.Y_PRED, ["mse"])
        assert set(values) == {"mse"}


class TestGenerateVideoFromFrames:
    @patch("ml_pic_collision_operators.test.subprocess.run")
    def test_builds_expected_ffmpeg_command(self, mock_run):
        _generate_video_from_frames(frame_dir="/frames", video_fname="out.mp4", fps=24)
        mock_run.assert_called_once()
        command = mock_run.call_args.args[0]
        assert command[0] == "ffmpeg"
        assert "24" in command  # fps threaded into -framerate and -r.
        assert "/frames/%06d.png" in command
        assert "out.mp4" in command
        assert mock_run.call_args.kwargs["check"] is True


class TestBuildTestFunctions:
    MONOMIAL_SPEC = TestFunctionConfig(
        cls_name="MonomialTestFunctions", cls_kwargs={"n_dims": 2, "degree": 2}
    )

    def test_single_spec_returned_unwrapped(self):
        tf = _build_test_functions([self.MONOMIAL_SPEC])
        assert isinstance(tf, MonomialTestFunctions)

    def test_multiple_specs_concatenated(self):
        tf = _build_test_functions([self.MONOMIAL_SPEC, self.MONOMIAL_SPEC])
        assert isinstance(tf, ConcatTestFunctions)


class TestHistogramFromTracks:
    BIN_RANGE = (0.0, 1.0, 0.0, 1.0)
    GRID_SIZE = (2, 2)

    def test_histogram_normalized_when_v_all_in_range(self):
        v = np.array([[0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]])
        h = _histogram_from_tracks(v, self.BIN_RANGE, self.GRID_SIZE)
        assert h.sum() == pytest.approx(1.0)

    def test_mass_drops_for_out_of_range_particles(self):
        # 3 of 4 particles inside the binned region, one well outside.
        v = np.array([[0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [5.0, 5.0]])
        h = _histogram_from_tracks(v, self.BIN_RANGE, self.GRID_SIZE)
        assert h.sum() == pytest.approx(0.75)

    def test_bad_bin_range_length_raises(self):
        v = np.zeros((4, 2))
        with pytest.raises(ValueError, match="bin_range"):
            _histogram_from_tracks(v, self.BIN_RANGE[:-1], self.GRID_SIZE)

    def test_bad_grid_size_length_raises(self):
        v = np.zeros((4, 2))
        with pytest.raises(ValueError, match="grid_size"):
            _histogram_from_tracks(v, self.BIN_RANGE, self.GRID_SIZE[:-1])
