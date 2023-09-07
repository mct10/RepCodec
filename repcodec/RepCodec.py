# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import torch.nn as nn

from repcodec.modules.decoder import Decoder
from repcodec.modules.encoder import Encoder
from repcodec.modules.projector import Projector
from repcodec.modules.quantizer import Quantizer


class RepCodec(nn.Module):
    def __init__(
            self,
            input_channels=768,
            output_channels=768,
            encode_channels=768,
            decode_channels=768,
            code_dim=768,
            codebook_num=1,
            codebook_size=1024,
            bias=True,
            enc_ratios=(1, 1),
            dec_ratios=(1, 1),
            enc_strides=(1, 1),
            dec_strides=(1, 1),
            enc_kernel_size=3,
            dec_kernel_size=3,
            enc_block_dilations=(1, 1),
            enc_block_kernel_size=3,
            dec_block_dilations=(1, 1),
            dec_block_kernel_size=3
    ):
        super().__init__()

        self.input_channels = input_channels

        self.encoder = Encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            channel_ratios=enc_ratios,
            strides=enc_strides,
            kernel_size=enc_kernel_size,
            bias=bias,
            block_dilations=enc_block_dilations,
            unit_kernel_size=enc_block_kernel_size
        )

        self.decoder = Decoder(
            code_dim=code_dim,
            output_channels=output_channels,
            decode_channels=decode_channels,
            channel_ratios=dec_ratios,
            strides=dec_strides,
            kernel_size=dec_kernel_size,
            bias=bias,
            block_dilations=dec_block_dilations,
            unit_kernel_size=dec_block_kernel_size
        )

        self.projector = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False
        )

        self.quantizer = Quantizer(
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.projector(x)
        zq, vqloss, perplexity = self.quantizer(z)
        y = self.decoder(zq)
        return y, zq, z, vqloss, perplexity
