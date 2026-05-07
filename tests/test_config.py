import pytest
from pydantic import ValidationError

from ml_pic_collision_operators.config.train import (
    TrainDataConfig,
    TemporalUnrolligStageConfig,
    ToggleWithFrequencyCallback,
    TrainCallbackConfig,
    LossConfig,
    TrainConfig,
)
from ml_pic_collision_operators.config.test import (
    MLflowModelConfig,
    HDFModelConfig,
    TestDataConfig,
    TestMetric,
    TestConfig,
)
from ml_pic_collision_operators.config.schema import MainConfig
from ml_pic_collision_operators.config.loader import load_config

_BASE_TRAIN = {
    "mode": "temporal_unrolling",
    "model_cls": "SomeModel",
    "data": {"folders": ["data/"]},
    "dataset_cls": "SomeDataset",
    "temporal_unrolling_stages": {"s1": {"unrolling_steps": 1, "epochs": 5}},
    "loss": {"name": "mse", "mode": "accumulated"},
}

_BASE_TEST = {
    "mode": "rollout",
    "data": {"folders": ["data/"]},
    "model": {"type": "hdf", "hdf_file": "model.hdf"},
}


class TestTrainDataConfig:
    def test_defaults(self):
        # Confirm the documented default values are wired up
        cfg = TrainDataConfig(folders=["data/"])
        assert cfg.train_valid_ratio == 1.0
        assert cfg.conditioners is None
        assert cfg.include_time is False

    def test_conditioners_matching_length_accepted(self):
        # The cross-field validator must pass when lengths are equal
        cfg = TrainDataConfig(folders=["a/", "b/"], conditioners=[{"k": 1}, {"k": 2}])
        assert len(cfg.conditioners) == 2

    def test_conditioners_length_mismatch_raises(self):
        # Custom model_validator must catch mismatched list lengths
        with pytest.raises(ValidationError, match="conditioners and folders"):
            TrainDataConfig(folders=["a/", "b/"], conditioners=[{"k": 1}])

    def test_conditioners_none_skips_length_check(self):
        # The validator has an explicit `if conditioners is not None` guard —
        # verify omitting conditioners never triggers the length check
        TrainDataConfig(folders=["a/", "b/", "c/"])

    def test_empty_folders_raises(self):
        # An empty folder list would cause a runtime error downstream
        with pytest.raises(ValidationError, match="folders must not be empty"):
            TrainDataConfig(folders=[])

    def test_train_valid_ratio_zero_raises(self):
        # ratio=0 means no training data; the open lower bound must be enforced
        with pytest.raises(ValidationError, match="train_valid_ratio"):
            TrainDataConfig(folders=["a/"], train_valid_ratio=0.0)

    def test_train_valid_ratio_negative_raises(self):
        with pytest.raises(ValidationError, match="train_valid_ratio"):
            TrainDataConfig(folders=["a/"], train_valid_ratio=-0.1)

    def test_train_valid_ratio_above_one_raises(self):
        # the upper bound is closed at 1.0; anything above must be rejected
        with pytest.raises(ValidationError, match="train_valid_ratio"):
            TrainDataConfig(folders=["a/"], train_valid_ratio=1.1)

    def test_train_valid_ratio_one_accepted(self):
        # 1.0 is a valid ratio (use all data for training, no validation split)
        cfg = TrainDataConfig(folders=["a/"], train_valid_ratio=1.0)
        assert cfg.train_valid_ratio == 1.0


class TestTemporalUnrolligStageConfig:
    def test_lr_defaults_to_none(self):
        # lr is optional; verify it doesn't default to 0 or some other sentinel
        cfg = TemporalUnrolligStageConfig(unrolling_steps=3, epochs=10)
        assert cfg.lr is None

    def test_explicit_lr_stored(self):
        cfg = TemporalUnrolligStageConfig(unrolling_steps=1, epochs=5, lr=1e-3)
        assert cfg.lr == pytest.approx(1e-3)


class TestToggleWithFrequencyCallback:
    def test_disabled_by_default(self):
        # Default state must not require a frequency
        cb = ToggleWithFrequencyCallback()
        assert cb.enabled is False
        assert cb.frequency is None

    def test_enabled_without_frequency_raises(self):
        # Custom model_validator: enabled=True without frequency must be caught
        with pytest.raises(ValidationError, match="frequency must be set"):
            ToggleWithFrequencyCallback(enabled=True)

    def test_disabled_with_frequency_is_valid(self):
        # Frequency is only checked when enabled=True; a disabled callback with a
        # pre-configured frequency should be silently accepted
        cb = ToggleWithFrequencyCallback(enabled=False, frequency=5)
        assert cb.frequency == 5


class TestTrainCallbackConfig:
    def test_non_trivial_defaults(self):
        # Several fields default to enabled=True with specific frequencies;
        # the rest default to enabled=False — verify the non-obvious ones explicitly
        cfg = TrainCallbackConfig()
        assert cfg.log_best_model.enabled is True
        assert cfg.log_best_model.frequency == "stage_end"
        assert cfg.log_metrics_epoch.enabled is True
        assert cfg.log_metrics_epoch.frequency == 1
        assert cfg.log_model.enabled is False
        assert cfg.log_metrics_step.enabled is False

    def test_log_best_model_rejects_invalid_frequency(self):
        # log_best_model is parameterised with Literal["always","stage_end","train_end"];
        # an integer frequency must be rejected even though ToggleWithFrequencyCallback
        # itself accepts any FrequencyType
        with pytest.raises(ValidationError):
            TrainCallbackConfig(
                log_best_model={"enabled": True, "frequency": 5}
            )

    def test_log_model_rejects_string_frequency(self):
        # log_model is parameterised with int; a Literal string must be rejected
        with pytest.raises(ValidationError):
            TrainCallbackConfig(
                log_model={"enabled": True, "frequency": "stage_end"}
            )


class TestLossConfig:
    def test_regularization_defaults_to_zero(self):
        # Both reg terms have non-obvious defaults; verify they're 0 not None
        cfg = LossConfig(name="mse", mode="last")
        assert cfg.reg_first_deriv == 0.0
        assert cfg.reg_second_deriv == 0.0


class TestTrainConfig:
    def test_defaults(self):
        # Confirm non-obvious defaults: random_seed, device, callbacks
        cfg = TrainConfig.model_validate(_BASE_TRAIN)
        assert cfg.random_seed == 42
        assert cfg.device == "cuda"
        assert cfg.callbacks is None

    def test_multiple_unrolling_stages(self):
        # temporal_unrolling_stages is an arbitrary-length dict — verify >1 entry works
        cfg_dict = {
            **_BASE_TRAIN,
            "temporal_unrolling_stages": {
                "warmup": {"unrolling_steps": 1, "epochs": 5},
                "main": {"unrolling_steps": 4, "epochs": 50},
            },
        }
        cfg = TrainConfig.model_validate(cfg_dict)
        assert len(cfg.temporal_unrolling_stages) == 2


class TestMLflowModelConfig:
    def test_change_params_defaults_to_none(self):
        # change_params is the only optional field — verify its default
        cfg = MLflowModelConfig(
            type="mlflow", experiment_name="exp", run_name="run", fname="model.pt"
        )
        assert cfg.change_params is None

    def test_change_params_accepted(self):
        # change_params is a free-form override dict; verify arbitrary keys are stored
        cfg = MLflowModelConfig(
            type="mlflow",
            experiment_name="exp",
            run_name="run",
            fname="model.pt",
            change_params={"lr": 1e-4},
        )
        assert cfg.change_params == {"lr": 1e-4}


class TestHDFModelConfig:
    def test_optional_fields_default_to_none(self):
        # Both params and change_params are optional; neither should default to {}
        cfg = HDFModelConfig(type="hdf", hdf_file="model.hdf")
        assert cfg.params is None
        assert cfg.change_params is None


class TestTestDataConfig:
    def test_defaults(self):
        # step_size replaces train_valid_ratio here; verify the correct default
        cfg = TestDataConfig(folders=["data/test"])
        assert cfg.step_size == 1
        assert cfg.conditioners is None

    def test_conditioners_length_mismatch_raises(self):
        # The same cross-field validator as TrainDataConfig must be present here too
        with pytest.raises(ValidationError, match="conditioners and folders"):
            TestDataConfig(folders=["a/"], conditioners=[{"k": 1}, {"k": 2}])

    def test_empty_folders_raises(self):
        with pytest.raises(ValidationError, match="folders must not be empty"):
            TestDataConfig(folders=[])

    def test_non_positive_step_size_raises(self):
        # step_size drives time integration; zero or negative has no physical meaning
        with pytest.raises(ValidationError, match="step_size must be positive"):
            TestDataConfig(folders=["a/"], step_size=0)

        with pytest.raises(ValidationError, match="step_size must be positive"):
            TestDataConfig(folders=["a/"], step_size=-1)


class TestTestMetricEnum:
    def test_all_members_round_trip(self):
        # Documents the full set of valid metric names; catches accidental renames
        for name in ("mse", "l1", "l2", "l1_norm", "l2_norm"):
            assert TestMetric(name).value == name


class TestTestConfig:
    def test_discriminated_union_resolves_hdf(self):
        # type="hdf" must produce HDFModelConfig, not the base union type
        cfg = TestConfig.model_validate(_BASE_TEST)
        assert isinstance(cfg.model, HDFModelConfig)

    def test_discriminated_union_resolves_mlflow(self):
        # type="mlflow" must produce MLflowModelConfig
        cfg_dict = {
            **_BASE_TEST,
            "model": {
                "type": "mlflow",
                "experiment_name": "exp",
                "run_name": "run",
                "fname": "model.pt",
            },
        }
        cfg = TestConfig.model_validate(cfg_dict)
        assert isinstance(cfg.model, MLflowModelConfig)

    def test_default_metrics_is_complete_set(self):
        # Default is set(TestMetric) — a non-obvious mutable default that Pydantic must copy
        cfg = TestConfig.model_validate(_BASE_TEST)
        assert cfg.metrics == set(TestMetric)

    def test_metrics_subset_via_strings(self):
        # Pydantic must coerce string values in the YAML to TestMetric enum members
        cfg_dict = {**_BASE_TEST, "metrics": ["mse", "l1"]}
        cfg = TestConfig.model_validate(cfg_dict)
        assert cfg.metrics == {TestMetric.mse, TestMetric.l1}


class TestMainConfig:
    def test_train_mode(self):
        cfg = MainConfig.model_validate({"mode": "train", "train": _BASE_TRAIN})
        assert isinstance(cfg.train, TrainConfig)
        assert cfg.test is None

    def test_test_mode(self):
        cfg = MainConfig.model_validate({"mode": "test", "test": _BASE_TEST})
        assert isinstance(cfg.test, TestConfig)
        assert cfg.train is None

    def test_train_mode_without_train_config_raises(self):
        # mode declares intent; omitting the matching sub-config is a silent misconfiguration
        with pytest.raises(ValidationError, match="train config must be provided"):
            MainConfig.model_validate({"mode": "train"})

    def test_test_mode_without_test_config_raises(self):
        with pytest.raises(ValidationError, match="test config must be provided"):
            MainConfig.model_validate({"mode": "test"})


class TestLoadConfig:
    def test_valid_yaml_returns_config_and_dict(self, tmp_path):
        # Smoke test: well-formed YAML must return a (MainConfig, raw dict) pair
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "mode: train\n"
            "train:\n"
            "  mode: temporal_unrolling\n"
            "  model_cls: SomeModel\n"
            "  data:\n"
            "    folders: ['data/']\n"
            "  dataset_cls: SomeDataset\n"
            "  temporal_unrolling_stages:\n"
            "    s1:\n"
            "      unrolling_steps: 1\n"
            "      epochs: 10\n"
            "  loss:\n"
            "    name: mse\n"
            "    mode: accumulated\n"
        )
        config, raw = load_config(str(cfg_file))
        assert isinstance(config, MainConfig)
        assert isinstance(raw, dict)

    def test_empty_yaml_raises(self, tmp_path):
        # yaml.safe_load returns None for empty files; load_config must raise explicitly
        # rather than letting model_validate receive None and produce a confusing error
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        with pytest.raises(Exception, match="Empty config file"):
            load_config(str(cfg_file))
