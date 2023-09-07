# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on fairseq (https://github.com/facebookresearch/fairseq)

import logging
import os
import sys

from feature_utils import get_path_iterator, dump_feature

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_feature")


def main(
        model_type: str,
        tsv_path: str,
        ckpt_path: str,
        whisper_root: str,
        whisper_name: str,
        layer: int,
        nshard: int,
        rank: int,
        feat_dir: str,
        max_chunk: int,
        use_cpu: bool = False
):
    device = "cpu" if use_cpu else "cuda"

    # some checks
    if model_type in ["hubert", "data2vec"]:
        assert ckpt_path and os.path.exists(ckpt_path)
    elif model_type in ["whisper"]:
        assert whisper_name and whisper_root
    else:
        raise ValueError(f"Unsupported model type {model_type}")

    reader = None
    if model_type == "hubert":
        from hubert_feature_reader import HubertFeatureReader
        reader = HubertFeatureReader(ckpt_path, layer, device=device, max_chunk=max_chunk)
    elif model_type == "data2vec":
        from data2vec_feature_reader import Data2vecFeatureReader
        reader = Data2vecFeatureReader(ckpt_path, layer, device=device, max_chunk=max_chunk)
    elif model_type == "whisper":
        from whisper_feature_reader import WhisperFeatureReader
        reader = WhisperFeatureReader(whisper_root, whisper_name, layer, device=device)

    assert reader is not None

    generator, num = get_path_iterator(tsv_path, nshard, rank)
    dump_feature(reader, generator, num, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        choices=["data2vec", "hubert", "whisper"],
        help="the type of the speech encoder."
    )
    parser.add_argument(
        "--tsv_path",
        required=True,
        type=str,
        help="the path to the tsv file."
    )
    parser.add_argument(
        "--ckpt_path",
        required=False,
        type=str,
        default=None,
        help="path to the speech model. must provide for HuBERT and data2vec"
    )
    parser.add_argument(
        "--whisper_root",
        required=False,
        type=str,
        default=None,
        help="root dir to download/store whisper model. must provide for whisper model."
    )
    parser.add_argument(
        "--whisper_name",
        required=False,
        type=str,
        default=None,
        help="name of whisper model. e.g., large-v2. must provide for whisper model."
    )
    parser.add_argument(
        "--layer",
        required=True,
        type=int,
        help="which layer of the model. this is 1-based."
    )
    parser.add_argument(
        "--feat_dir",
        required=True,
        type=str,
        help="the output dir to save the representations."
    )
    parser.add_argument(
        "--nshard",
        required=False,
        type=int,
        default=1,
        help="total number of shards."
    )
    parser.add_argument(
        "--rank",
        required=False,
        type=int,
        default=0,
        help="shard id of this process."
    )
    parser.add_argument(
        "--max_chunk",
        type=int,
        default=1600000,
        help="max number of frames of each batch."
    )
    parser.add_argument(
        "--use_cpu",
        default=False,
        action="store_true",
        help="whether use cpu instead of gpu."
    )
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
