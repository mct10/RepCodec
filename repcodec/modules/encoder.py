# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import torch
import torch.nn as nn

from repcodec.layers.conv_layer import Conv1d
from repcodec.modules.residual_unit import ResidualUnit


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            dilations=(1, 1),
            unit_kernel_size=3,
            bias=True
    ):
        super().__init__()
        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [
                ResidualUnit(in_channels, in_channels,
                             kernel_size=unit_kernel_size,
                             dilation=dilation)
            ]
        self.num_res = len(self.res_units)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3 if stride == 1 else (2 * stride),  # special case: stride=1, do not use kernel=2
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            encode_channels: int,
            channel_ratios=(1, 1),
            strides=(1, 1),
            kernel_size=3,
            bias=True,
            block_dilations=(1, 1),
            unit_kernel_size=3
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv = Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False
        )
        self.conv_blocks = torch.nn.ModuleList()
        in_channels = encode_channels
        for idx, stride in enumerate(strides):
            out_channels = int(encode_channels * channel_ratios[idx])  # could be float
            self.conv_blocks += [
                EncoderBlock(in_channels, out_channels, stride,
                             dilations=block_dilations, unit_kernel_size=unit_kernel_size,
                             bias=bias)
            ]
            in_channels = out_channels
        self.num_blocks = len(self.conv_blocks)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        return x
