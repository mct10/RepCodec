# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on fairseq (https://github.com/facebookresearch/fairseq) and
# Whisper (https://github.com/openai/whisper/)

import io
import logging
import os
from typing import Optional, Union

import soundfile as sf
import torch
from whisper import _MODELS, _download, _ALIGNMENT_HEADS, available_models
from whisper.audio import log_mel_spectrogram
from whisper.model import ModelDimensions

from whisper_model import Whisper_

logger = logging.getLogger("dump_feature")


def load_model(
        name: str,
        device: Optional[Union[str, torch.device]] = None,
        download_root: str = None,
        in_memory: bool = False,
) -> Whisper_:
    """
    Reference: https://github.com/openai/whisper/blob/main/whisper/__init__.py#L97
    But we will load a `Whisper_` model for feature extraction.

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
            io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper_(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)


class WhisperFeatureReader(object):
    def __init__(self, root, ckpt, layer, device):
        self.device = device
        logger.info(f"device = {self.device}")

        self.model: Whisper_ = load_model(name=ckpt, device=self.device, download_root=root).eval()
        self.model.decoder = None  # to save some memory by deleting the decoder
        self.layer = layer  # one-based

    def read_audio(self, path, ref_len=None):
        wav, sample_rate = sf.read(path)
        assert sample_rate == 16000, sample_rate
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logger.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        wav = self.read_audio(path, ref_len)
        audio_length = len(wav)
        with torch.no_grad():
            mel = log_mel_spectrogram(torch.from_numpy(wav).float().to(self.device))
            hidden = self.model.extract_features(mel.unsqueeze(0), target_layer=self.layer)
            feature_length = audio_length // 320
            hidden = hidden[0, :feature_length]
        return hidden.contiguous()
