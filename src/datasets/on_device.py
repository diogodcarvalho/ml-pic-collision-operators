import torch
from .temporal_unrolled import TemporalUnrolledDataset


class OnDeviceDataset(TemporalUnrolledDataset):

    def __init__(
        self,
        folder: str,
        i_start: int = 0,
        i_end: int = -1,
        step_size: int = 1,
        temporal_unroll_steps: int = 1,
        device: str = "cuda",
    ):
        super().__init__(
            folder=folder,
            i_start=i_start,
            i_end=i_end,
            step_size=step_size,
            temporal_unroll_steps=temporal_unroll_steps,
        )
        self.device = device

        self.input_data = torch.stack(
            [torch.Tensor(self._load_inputs(idx)) for idx in range(len(self))],
            dim=0,
        ).to(device)
        self.target_data = torch.stack(
            [torch.Tensor(self._load_targets(idx)) for idx in range(len(self))],
            dim=0,
        ).to(device)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        inputs = self.input_data[idx]
        targets = self.target_data[idx]
        return inputs, targets, self.dt
