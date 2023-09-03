# RepCodec: A Speech Representation Codec for Speech Tokenization

> [**RepCodec: A Speech Representation Codec for Speech Tokenization**]()

## Introduction

**RepCodec** is a speech tokenization method for converting a speech waveform into a sequence of discrete semantic tokens.
The main idea is to train a representation codec which learns a vector quantization codebook through reconstructing the
input speech representations from speech encoders like HuBERT or data2vec.
Extensive experiments show that RepCodec significantly outperforms the widely used k-means clustering approach in both
speech understanding and generation.
Also, RepCodec generalizes well across various speech encoders and languages.

<img src="images/RepCodec.png" alt="se" width="1000" />

## Pre-Trained Models

| Feature Type              | Speech Data                                              | Model        |
|---------------------------|----------------------------------------------------------|--------------|
| HuBERT base layer 9       | [Librispeech](http://www.openslr.org/12) train-clean-100 | [download]() |
| HuBERT large layer 18     | [Librispeech](http://www.openslr.org/12) train-clean-100 | [download]() |
| data2vec base layer 6     | [Librispeech](http://www.openslr.org/12) train-clean-100 | [download]() |
| data2vec large layer 18   | [Librispeech](http://www.openslr.org/12) train-clean-100 | [download]() |
| Whisper medium layer 24   | [Librispeech](http://www.openslr.org/12) train-clean-100 | [download]() |
| Whisper large-v2 layer 32 | [Librispeech](http://www.openslr.org/12) train-clean-100 | [download]() |

## Speech Tokenization Using Pre-Trained Models

```python
import torch
import yaml

from model.RepCodec import RepCodec

# for feature types of HubERT base & data2vec base, please use repcodec_dim768.yaml;
# for feature types of HuBERT large & data2vec large & Whisper medium, please use repcodec_dim1024.yaml;
# for feature types of Whisper large-v2, please use repcodec_dim1280.yaml
config = "./configs/repcodec_dim768.yaml"
with open(config) as fp:
    conf = yaml.load(fp, Loader=yaml.FullLoader)

model = RepCodec(**conf)
model.load_state_dict(torch.load("./hubert_base_l9.pkl", map_location="cpu")["model"]["repcodec"])
model.quantizer.initial()
model.eval()

# input shape: (batch size, hidden dim, sequence length)
random_features = torch.randn(size=(1, 768, 100))
with torch.no_grad():
    x = model.encoder(random_features)
    z = model.projector(x)
    _, idx = model.quantizer.codebook.forward_index(z.transpose(2, 1))
    tokens = idx.cpu().data.numpy().tolist()[0]
```

## Acknowledge
Our implementation is based on [facebookresearch/AudioDec](https://github.com/facebookresearch/AudioDec).
We thank them for open-sourcing their code!

## Citation
If you find our work useful, please cite the following article.
```

```