import torch
from torch.utils.data import DataLoader, default_collate
from dataclasses import dataclass

from ml_pic_collision_operators.datasets.base import DatasetItem


@dataclass
class BatchDatasetItem:
    inputs: torch.Tensor
    targets: torch.Tensor
    dt: torch.Tensor
    conditioners: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        return self.dt.shape[0]

    # required custom memory pinning method on custom type
    def pin_memory(self):
        self.inputs = self.inputs.pin_memory()
        self.targets = self.targets.pin_memory()
        self.dt = self.dt.pin_memory()
        if self.conditioners is not None:
            self.conditioners = self.conditioners.pin_memory()
        return self

    def to_device(self, device: torch.device) -> "BatchDatasetItem":
        self.inputs = self.inputs.to(device, non_blocking=True)
        self.targets = self.targets.to(device, non_blocking=True)
        self.dt = self.dt.to(device, non_blocking=True)
        if self.conditioners is not None:
            self.conditioners = self.conditioners.to(device, non_blocking=True)
        return self


def collate_fn(batch: list[DatasetItem]) -> BatchDatasetItem:
    """Collate function to convert a list of DatasetItem into a BatchDatasetItem."""

    def _prepare(name: str) -> torch.Tensor:
        items = [getattr(b, name) for b in batch]
        collated = default_collate(items)
        return collated.to(torch.get_default_dtype())

    return BatchDatasetItem(
        inputs=_prepare("inputs"),
        targets=_prepare("targets"),
        dt=_prepare("dt"),
        conditioners=(
            _prepare("conditioners") if batch[0].conditioners is not None else None
        ),
    )


class BaseDataLoader(DataLoader):

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=None,
        persistent_workers=False,
        device=None,
    ):
        self.device = device
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def __iter__(self):
        for batch in super().__iter__():
            if self.device is not None:
                yield batch.to_device(self.device)
            else:
                yield batch
