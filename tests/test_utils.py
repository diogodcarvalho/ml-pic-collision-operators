import os
import pytest
from unittest.mock import patch

from ml_pic_collision_operators.utils import (
    class_from_str,
    is_distributed,
    setup_distributed,
    cleanup_ddp,
    root_print,
    rank_print,
)


class TestClassFromStr:
    def test_load_torch_optimizer(self):
        cls = class_from_str("torch.optim.Adam")
        from torch.optim import Adam

        assert cls == Adam

    def test_load_dataset_class(self):
        cls = class_from_str("BaseDataset", "ml_pic_collision_operators.datasets")
        from ml_pic_collision_operators.datasets import BaseDataset

        assert cls == BaseDataset

    def test_load_model_class(self):
        cls = class_from_str(
            "FokkerPlanck2D_Tensor_AD", "ml_pic_collision_operators.models"
        )
        from ml_pic_collision_operators.models import FokkerPlanck2D_Tensor_AD

        assert cls == FokkerPlanck2D_Tensor_AD

    def test_invalid_module(self):
        with pytest.raises(ImportError):
            class_from_str("FokkerPlanck2D_Tensor_AD", "non_existent_module")

    def test_invalid_class(self):
        with pytest.raises(AttributeError):
            class_from_str("NonExistentClass", "ml_pic_collision_operators.models")


class TestDistributedChecks:
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}, clear=True)
    def test_is_distributed_true(self):
        assert is_distributed() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_distributed_false(self):
        assert is_distributed() is False


class TestSetupCleanupDDP:
    @patch("torch.cuda.set_device")
    @patch("torch.distributed.init_process_group")
    @patch.dict(
        os.environ, {"RANK": "1", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}, clear=True
    )
    def test_setup_distributed_multi_node(self, mock_init, mock_set_device):
        rank, local_rank, world_size = setup_distributed()
        assert rank == 1
        assert local_rank == 0
        assert world_size == 2
        mock_init.assert_called_once_with(backend="nccl", rank=1, world_size=2)
        mock_set_device.assert_called_once_with(0)

    @patch("torch.cuda.set_device")
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_distributed_single_node(self, mock_set_device):
        rank, local_rank, world_size = setup_distributed()
        assert rank == 0
        assert world_size == 1
        mock_set_device.assert_called_once_with(0)

    @patch("torch.distributed.destroy_process_group")
    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}, clear=True)
    def test_cleanup_ddp(self, mock_destroy):
        cleanup_ddp()
        mock_destroy.assert_called_once()


class TestPrintDDP:
    @patch("builtins.print")
    @patch("torch.distributed.get_rank", return_value=3)
    def test_rank_print(self, mock_get_rank, mock_print):
        rank_print("hello")
        mock_print.assert_called_with("[Rank 3]", "hello")

    @patch("builtins.print")
    @patch("ml_pic_collision_operators.utils.is_distributed", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_root_print_on_rank_0(self, mock_get_rank, mock_is_dist, mock_print):
        root_print("secret message")
        mock_print.assert_called_once_with("secret message")

    @patch("builtins.print")
    @patch("ml_pic_collision_operators.utils.is_distributed", return_value=True)
    @patch("torch.distributed.get_rank", return_value=1)
    def test_root_print_on_other_rank(self, mock_get_rank, mock_is_dist, mock_print):
        root_print("secret message")
        mock_print.assert_not_called()
