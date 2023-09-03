import argparse
import os.path

import numpy as np
import torch
import yaml

from RepCodec import RepCodec

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
        "representation",
        type=str,
        help="representation to be tokenized. the shape should be (sequence len, hidden dim)."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=list(ALL_MODELS.keys()),
        help="name of the RepCodec model"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="the directory to store the model files."
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        action="store_true",
        help="whether use gpu for inference."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="the directory to save the output."
    )
    return parser.parse_args()


def load_model(name: str, model_dir: str):
    config = os.path.join(os.path.dirname(__file__), "../configs", f"repcodec_dim{ALL_MODELS[name]}.yaml")
    with open(config) as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)
    model = RepCodec(**conf)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{name}.pkl"), map_location="cpu")["model"]["repcodec"])
    model.quantizer.initial()
    model.eval()
    return model


def cli():
    args = parse_args()

    input_file = args.representation
    model = load_model(name=args.model, model_dir=args.model_dir)

    data = np.load(input_file)
    data = torch.tensor(data, dtype=torch.float).unsqueeze(0).transpose(1, 2)

    device = "cuda" if args.use_gpu else "cpu"
    model.to(device)
    data.to(device)
    with torch.no_grad():
        x = model.encoder(data)
        z = model.projector(x)
        _, idx = model.quantizer.codebook.forward_index(z.transpose(2, 1))
        tokens = idx.cpu().data.numpy().tolist()[0]

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{os.path.basename(input_file)}.tokens"), mode="w+") as fp:
        fp.write(f"{' '.join(tokens)}\n")


if __name__ == '__main__':
    cli()
