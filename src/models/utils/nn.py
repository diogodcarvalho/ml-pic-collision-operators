import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, in_size, out_size, depth, width_size, activation, use_bias, use_final_bias
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_size, width_size, bias=use_bias))
            # layers.append(nn.BatchNorm1d(width_size))
            layers.append(activation())
            in_size = width_size
        layers.append(nn.Linear(in_size, out_size, bias=use_final_bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
