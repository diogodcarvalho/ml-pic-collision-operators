import torch
import numpy as np
import pytest

from ml_pic_collision_operators.dataloaders.base import BaseDataLoader, BatchDatasetItem
from ml_pic_collision_operators.datasets.base import DatasetItem


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def test_base_dataloader_returns_dataset_item_batch():
    dataset = DummyDataset(
        [
            DatasetItem(
                inputs=np.array([1.0, 2.0], dtype=np.float32),
                targets=np.array([3.0], dtype=np.float32),
                dt=0.1,
            ),
            DatasetItem(
                inputs=np.array([4.0, 5.0], dtype=np.float32),
                targets=np.array([6.0], dtype=np.float32),
                dt=0.2,
            ),
            DatasetItem(
                inputs=np.array([7.0, 8.0], dtype=np.float32),
                targets=np.array([9.0], dtype=np.float32),
                dt=0.3,
            ),
        ]
    )
    loader = BaseDataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))

    assert isinstance(batch, BatchDatasetItem)
    assert torch.is_tensor(batch.inputs)
    assert torch.is_tensor(batch.targets)
    assert torch.is_tensor(batch.dt)
    assert batch.conditioners is None
    assert batch.inputs.shape == (2, 2)
    assert batch.targets.shape == (2, 1)
    assert batch.dt.shape == (2,)
    assert np.allclose(batch.inputs.tolist(), [[1.0, 2.0], [4.0, 5.0]])
    assert np.allclose(batch.targets.tolist(), [[3.0], [6.0]])
    assert np.allclose(batch.dt.tolist(), [0.1, 0.2])


def test_base_dataloader_preserves_conditioners_when_present():
    dataset = DummyDataset(
        [
            DatasetItem(
                inputs=np.array([1.0, 2.0], dtype=np.float32),
                targets=np.array([3.0], dtype=np.float32),
                dt=np.array([0.1]),
                conditioners=np.array([10.0], dtype=np.float32),
            ),
            DatasetItem(
                inputs=np.array([4.0, 5.0], dtype=np.float32),
                targets=np.array([6.0], dtype=np.float32),
                dt=np.array([0.2]),
                conditioners=np.array([20.0], dtype=np.float32),
            ),
        ]
    )
    loader = BaseDataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert isinstance(batch, BatchDatasetItem)
    assert torch.is_tensor(batch.conditioners)
    assert batch.conditioners.shape == (2, 1)
    assert np.allclose(batch.conditioners.tolist(), [[10.0], [20.0]])


def test_base_dataloader_casts_float_tensors_to_default_dtype():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dataset = DummyDataset(
            [
                DatasetItem(
                    inputs=torch.tensor([1.0, 2.0], dtype=torch.float32),
                    targets=torch.tensor([3.0], dtype=torch.float32),
                    dt=np.array([0.1]),
                ),
                DatasetItem(
                    inputs=torch.tensor([4.0, 5.0], dtype=torch.float32),
                    targets=torch.tensor([6.0], dtype=torch.float32),
                    dt=np.array([0.2]),
                ),
            ]
        )
        loader = BaseDataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert batch.inputs.dtype == torch.float64
        assert batch.targets.dtype == torch.float64
        assert batch.dt.dtype == torch.float64
    finally:
        torch.set_default_dtype(previous_dtype)


def test_batch_dataset_item_batch_size():
    batch = BatchDatasetItem(
        inputs=torch.tensor([[1.0, 2.0], [4.0, 5.0]]),
        targets=torch.tensor([[3.0], [6.0]]),
        dt=torch.tensor([0.1, 0.2]),
        conditioners=None,
    )
    assert batch.batch_size == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("conditioners", [None, torch.tensor([[10.0], [20.0]])])
def test_batch_dataset_item_to_device(conditioners):
    batch = BatchDatasetItem(
        inputs=torch.tensor([[1.0, 2.0], [4.0, 5.0]]),
        targets=torch.tensor([[3.0], [6.0]]),
        dt=torch.tensor([0.1, 0.2]),
        conditioners=conditioners,
    )
    result = batch.to_device(torch.device("cuda"))
    assert result is batch
    assert batch.inputs.device.type == "cuda"
    assert batch.targets.device.type == "cuda"
    assert batch.dt.device.type == "cuda"
    if conditioners is not None:
        assert batch.conditioners.device.type == "cuda"
    else:
        assert batch.conditioners is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("conditioners", [None, torch.tensor([[10.0]])])
def test_batch_dataset_item_pin_memory(conditioners):
    batch = BatchDatasetItem(
        inputs=torch.tensor([[1.0, 2.0]]),
        targets=torch.tensor([[3.0]]),
        dt=torch.tensor([0.1]),
        conditioners=conditioners,
    )
    result = batch.pin_memory()
    assert result is batch
    assert batch.inputs.is_pinned()
    assert batch.targets.is_pinned()
    assert batch.dt.is_pinned()
    if conditioners is not None:
        assert batch.conditioners.is_pinned()
    else:
        assert batch.conditioners is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("conditioners", [None, np.array([10.0], dtype=np.float32)])
def test_base_dataloader_device_moves_batches(conditioners):
    dataset = DummyDataset(
        [
            DatasetItem(
                inputs=np.array([1.0, 2.0], dtype=np.float32),
                targets=np.array([3.0], dtype=np.float32),
                dt=0.1,
                conditioners=conditioners,
            ),
        ]
    )
    loader = BaseDataLoader(dataset, batch_size=1, shuffle=False, device=torch.device("cuda"))
    batch = next(iter(loader))
    assert batch.inputs.device.type == "cuda"
    assert batch.targets.device.type == "cuda"
    assert batch.dt.device.type == "cuda"
    if conditioners is not None:
        assert batch.conditioners.device.type == "cuda"
    else:
        assert batch.conditioners is None
