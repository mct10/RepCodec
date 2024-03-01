# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import torch.nn as nn


class ReprReconstructLoss(nn.Module):
    def __init__(self, loss_type: str):
        super().__init__()
        if loss_type.lower() == "l1":
            self.loss_metric = nn.L1Loss()
        elif loss_type.lower() == "l2":
            self.loss_metric = nn.MSELoss()
        else:
            raise NotImplementedError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred, target):
        return self.loss_metric(pred, target)
