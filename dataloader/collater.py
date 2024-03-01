# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import numpy as np
import torch


class ReprCollater(object):
    def __call__(self, batch):
        xs = []
        for b in batch:
            if b is not None:
                xs.append(b)

        x_batch = np.stack(xs, axis=0)
        x_batch = torch.tensor(x_batch, dtype=torch.float).transpose(1, 2)  # (B, T, C) -> (B, C, T)

        return x_batch
