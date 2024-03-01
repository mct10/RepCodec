# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from repcodec.RepCodec import RepCodec

ALL_MODELS = {
    "data2vec_base_l6": 768,
    "data2vec_large_l18": 1024,
    "hubert_base_l9": 768,
    "hubert_large_l18": 1024,
    "whisper_medium_l24": 1024,
    "whisper_large_l32": 1280
}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "in_dir",
        type=str,
        help="directory of representations to be tokenized."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="path of the RepCodec model."
    )
    parser.add_argument(
        "--tsv_path",
        required=True,
        type=str,
        help="path of the tsv file."
    )
    parser.add_argument(
        "--model_config_path",
        default=None,
        type=str,
        help="please provide this training config if you are using the model you trained yourself."
    )
    parser.add_argument(
        "--n_shard",
        required=False,
        type=int,
        default=1,
        help="number of shards of representations."
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        action="store_true",
        help="whether use gpu for inference."
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="number of utterances for each mini batch."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="the directory to save the output."
    )
    return parser.parse_args()


def load_model(model_path: str, config_path: Optional[str] = None):
    if config_path is None:
        name = os.path.basename(model_path).strip(".pkl")
        assert name in ALL_MODELS.keys(), f"Cannot find configs for {model_path}. " \
                                          f"Please provide the config file you used for training."
        config = os.path.join(os.path.dirname(__file__), "configs", f"repcodec_dim{ALL_MODELS[name]}.yaml")
        with open(config) as fp:
            conf = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        with open(config_path) as fp:
            conf = yaml.load(fp, Loader=yaml.FullLoader)["model_params"]

    model = RepCodec(**conf)
    model.load_state_dict(torch.load(model_path, map_location="cpu")["model"]["repcodec"])
    model.quantizer.initial()
    model.eval()
    return model


def load_shard(in_dir: Path, rank: int, n_shard: int) -> Tuple[np.ndarray, List[int]]:
    feat_path = in_dir / f"{rank}_{n_shard}.npy"
    len_path = in_dir / f"{rank}_{n_shard}.len"

    with open(len_path) as fp:
        lengths = [int(line.strip()) for line in fp]

    return np.load(feat_path.as_posix(), mmap_mode="r"), lengths


def pad_data(data: List[np.ndarray]) -> List[np.ndarray]:
    max_len = max([d.shape[0] for d in data])
    data = [
        np.pad(d, [(0, max_len - d.shape[0]), (0, 0)], "constant", constant_values=0.0)
        for d in data
    ]
    return data


def make_batch_data(data: np.ndarray, shard_lengths: List[int], batch_size: int):
    batch_data = []
    batch_lens = []
    offsets = np.cumsum([0] + shard_lengths)
    assert len(data) == offsets[-1], f"{len(data)} {offsets[-1]}"

    # from longest to shortest
    for i in range(len(shard_lengths)):
        if batch_size > len(batch_data):
            batch_data.append(data[offsets[i]: offsets[i + 1]])
            batch_lens.append(shard_lengths[i])
        else:
            yield {
                "data": torch.tensor(np.stack(pad_data(batch_data)), dtype=torch.float),  # (bsz, seq len, hidden dim)
                "lengths": batch_lens
            }
            batch_data = [data[offsets[i]: offsets[i + 1]]]
            batch_lens = [shard_lengths[i]]
    if len(batch_data) > 0:
        yield {
            "data": torch.tensor(np.stack(pad_data(batch_data)), dtype=torch.float),
            "lengths": batch_lens
        }


def tokenize_batch(model: RepCodec, batch: dict, device: str) -> List[List[int]]:
    with torch.no_grad():
        data = batch["data"].transpose(1, 2).to(device)  # (bsz, hidden dim, seq len)
        x = model.encoder(data)
        z = model.projector(x)
        _, idx = model.quantizer.codebook.forward_index(z.transpose(2, 1))

    # when bsz=1: (1, seq len)
    if idx.dim() == 2:
        return idx.cpu().data.numpy().tolist()
    # when bsz>1: (1, bsz, seq len)
    tokens = idx.cpu().data.numpy().tolist()[0]
    res = []
    batch_lens = batch["lengths"]
    for i in range(len(tokens)):
        n_tokens = batch_lens[i]
        res.append(tokens[i][:n_tokens])
    return res


def load_tsv(path: str):
    with open(path) as fp:
        root = fp.readline().strip()
        names = []
        for line in fp:
            names.append(line.strip().split("\t")[0])
    return root, names


def cli():
    args = parse_args()
    device = "cuda" if args.use_gpu else "cpu"

    model = load_model(model_path=args.model, config_path=args.model_config_path)
    model.to(device)

    in_dir = Path(args.in_dir)
    n_shard = args.n_shard
    batch_size = args.batch_size

    root_dir, file_names = load_tsv(args.tsv_path)

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    processed_cnt = 0
    pbar = tqdm(total=len(file_names))
    with open(os.path.join(output_dir, "tokens"), mode="w+") as fp:
        fp.write(f"{root_dir}\n")

        for rank in range(n_shard):
            shard_data, shard_lengths = load_shard(in_dir, rank, n_shard)
            for batch in make_batch_data(shard_data, shard_lengths, batch_size=batch_size):
                batch_tokens = tokenize_batch(model, batch, device)

                for tokens in batch_tokens:
                    fp.write(f"{file_names[processed_cnt]}\t{' '.join(map(str, tokens))}\n")
                    processed_cnt += 1

                pbar.update(len(batch_tokens))
    assert processed_cnt == len(file_names), f"# lines of tsv do not match # of representations!"

    pbar.close()
    print("Tokenize successfully!")


if __name__ == '__main__':
    cli()
