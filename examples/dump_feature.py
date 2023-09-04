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
        layer: int,
        nshard: int,
        rank: int,
        feat_dir: str,
        max_chunk: int
):
    if model_type == "hubert":
        from hubert_feature_reader import HubertFeatureReader
        reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    elif model_type == "data2vec":
        from data2vec_feature_reader import Data2vecFeatureReader
        reader = Data2vecFeatureReader(ckpt_path, layer, max_chunk)
    else:
        raise ValueError(f"Unsupported model type {model_type}")

    generator, num = get_path_iterator(tsv_path, nshard, rank)
    dump_feature(reader, generator, num, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type",
        type=str,
        choices=["data2vec", "hubert", "whisper"],
        help="the type of the speech encoder."
    )
    parser.add_argument(
        "tsv_path",
        type=str,
        help="the path to the tsv file."
    )
    parser.add_argument(
        "ckpt_path",
        type=str,
        help="path to the speech model."
    )
    parser.add_argument(
        "layer",
        type=int,
        help="which layer of the model. should be 1-based."
    )
    parser.add_argument(
        "nshard",
        type=int,
        help="total number of shards."
    )
    parser.add_argument(
        "rank",
        type=int,
        help="shard id of this process."
    )
    parser.add_argument(
        "feat_dir",
        type=str,
        help="the output dir to save the features."
    )
    parser.add_argument(
        "--max_chunk",
        type=int,
        default=1600000,
        help="max number of frames of each batch."
    )
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
