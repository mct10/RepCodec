# RepCodec: A Speech Representation Codec for Speech Tokenization

Code and models will be released soon.

[[Paper]() | [Website]()]

## Introduction
RepCodec is a speech tokenization method for converting a speech waveform into a sequence of discrete semantic tokens.
The main idea is to train a representation codec which learns a vector quantization codebook through reconstructing the input speech representations from speech encoders like HuBERT or data2vec.
Extensive experiments show that RepCodec significantly outperforms the widely used k-means clustering approach in both speech understanding and generation.
Also, RepCodec generalizes well across various speech encoders and languages.

<img src="images/RepCodec.png" alt="se" width="1000" />
