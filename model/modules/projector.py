import torch.nn as nn

from model.layers.conv_layer import Conv1d


class Projector(nn.Module):
    def __init__(
            self,
            input_channels: int,
            code_dim: int,
            kernel_size=3,
            stride=1,
            bias=False
    ):
        super().__init__()
        self.project = Conv1d(
            input_channels,
            code_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

    def forward(self, x):
        return self.project(x)