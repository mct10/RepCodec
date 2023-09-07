# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on fairseq (https://github.com/facebookresearch/fairseq)

import logging

import torch
import torch.nn.functional as F
from fairseq import tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.data.audio.audio_utils import get_features_or_waveform
from omegaconf import OmegaConf

from data2vec_audio import Data2VecAudioModel

logger = logging.getLogger("dump_feature")


class Data2vecFeatureReader(object):
    def __init__(self, ckpt_path: str, layer: int, device: str, max_chunk=1600000):
        state = load_checkpoint_to_cpu(ckpt_path)
        cfg = state["cfg"]
        # load task
        task = tasks.setup_task(cfg.task, from_checkpoint=True)
        task.load_state_dict(state["task_state"])
        # load model config
        if "layer_type" not in cfg.model:
            # fix a missing key
            model_config = {k: v for k, v in cfg.model.items()}
            model_config["layer_type"] = "transformer"
            model_config = OmegaConf.create(model_config)
        else:
            model_config = cfg.model

        # fix param name in the state
        state["model"]["final_proj.weight"] = state["model"].pop("final_proj.0.weight")
        state["model"]["final_proj.bias"] = state["model"].pop("final_proj.0.bias")
        del state["model"]["_ema"]

        # load model
        model = Data2VecAudioModel.build_model(model_config)
        model.load_state_dict(
            state["model"], strict=True, model_cfg=model_config
        )

        self.device = device
        logger.info(f"device = {self.device}")

        self.model = model.eval().to(self.device)
        self.task = task
        self.layer = layer - 1  # make it 1-based
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logger.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                res = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    layer=self.layer,
                )
                feat_chunk = res["x"]
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)
