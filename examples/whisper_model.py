# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on fairseq (https://github.com/facebookresearch/fairseq) and 
# Whisper (https://github.com/openai/whisper/)

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from whisper.model import AudioEncoder, sinusoids, Whisper, ModelDimensions


class AudioEncoder_(AudioEncoder):
    def __init__(self, *args, **kwargs):
        super(AudioEncoder_, self).__init__(*args, **kwargs)

    def extract_feature(self, x: Tensor, target_layer: Optional[int] = None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        length_x = x.shape[1]
        if length_x > self.positional_embedding.shape[0]:
            self.register_buffer("positional_embedding", sinusoids(length_x, self.positional_embedding.shape[1]))
            self.positional_embedding = self.positional_embedding.to(x.device)
        x = (x + self.positional_embedding[:length_x, :]).to(x.dtype)

        if target_layer is None:
            target_layer = len(self.blocks)

        for block in self.blocks[:target_layer]:
            x = block(x)

        return x


class Whisper_(Whisper):
    def __init__(self, dims: ModelDimensions):
        super(Whisper_, self).__init__(dims)
        # replace audio encoder with our audio encoder
        self.encoder = AudioEncoder_(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )

    def extract_features(self, mel: torch.Tensor, target_layer: Optional[int] = None):
        return self.encoder.extract_feature(mel, target_layer)
