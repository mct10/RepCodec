import torch.nn as nn

from model.layers.vq_module import ResidualVQ


class Quantizer(nn.Module):
    def __init__(
            self,
            code_dim: int,
            codebook_num: int,
            codebook_size: int,
    ):
        super().__init__()
        self.codebook = ResidualVQ(
            dim=code_dim,
            num_quantizers=codebook_num,
            codebook_size=codebook_size
        )

    def initial(self):
        self.codebook.initial()

    def forward(self, z):
        zq, vqloss, perplexity = self.codebook(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, vqloss, perplexity

    def inference(self, z):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, indices

    def encode(self, z):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1), flatten_idx=True)
        return zq, indices

    def decode(self, indices):
        z = self.codebook.lookup(indices)
        return z
