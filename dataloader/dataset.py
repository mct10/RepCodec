# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import glob
import logging
import os
from typing import List

import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
logger = logging.getLogger("dataset")


class ReprDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            batch_len: int,
    ):
        self.batch_len = batch_len

        self.blocks = self._load_blocks(data_dir)
        self.offsets = self._load_offsets(data_dir)
        assert len(self.blocks) == len(self.offsets)
        # check len
        for i in range(len(self.blocks)):
            assert self.blocks[i].shape[0] == self.offsets[i][-1]

        self.n_examples = np.cumsum([0] + [offset.shape[0] - 1 for offset in self.offsets])

    def __len__(self):
        return self.n_examples[-1]

    def __getitem__(self, idx):
        # find which block
        block_id = -1
        for n in range(len(self.n_examples) - 1):
            if self.n_examples[n] <= idx < self.n_examples[n + 1]:
                block_id = n
                break
        assert 0 <= block_id < len(self.blocks), f"Failed to find {idx}"
        block_offset = idx - self.n_examples[block_id]
        start = self.offsets[block_id][block_offset]
        end = self.offsets[block_id][block_offset + 1]

        # randomly choose a slice
        if end - start < self.batch_len:
            return None
        elif end - start == self.batch_len:
            return self.blocks[block_id][start:end]
        else:
            start_offset = np.random.randint(low=start, high=end - self.batch_len)
            return self.blocks[block_id][start_offset:start_offset + self.batch_len]

    @staticmethod
    def _load_blocks(feat_dir: str) -> List[np.ndarray]:
        # e.g., 0_2.npy, 1_2.npy
        file_names = glob.glob(os.path.join(feat_dir, "*.npy"), recursive=False)
        # sort by index
        file_names = sorted(file_names, key=lambda x: int(os.path.basename(x).split("_")[0]))
        logger.info(f"Found following blocks: {file_names}")
        blocks = [np.load(name, mmap_mode="r") for name in file_names]
        return blocks

    @staticmethod
    def _load_offsets(feat_dir: str):
        def load_lens(file_name: str):
            with open(file_name, mode="r") as fp:
                res = fp.read().strip().split("\n")
            # for easy use. [res[i], res[i+1]) denotes the range for ith element
            res = [0] + [int(r) for r in res]
            return np.cumsum(res, dtype=int)

        # e.g., 0_2.len, 1_2.len
        file_names = glob.glob(os.path.join(feat_dir, "*.len"), recursive=False)
        file_names = sorted(file_names, key=lambda x: int(os.path.basename(x).split("_")[0]))
        file_lens = []
        for name in file_names:
            file_lens.append(load_lens(name))
        return file_lens
