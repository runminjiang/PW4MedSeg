
import torch
import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock
# from monai.networks.blocks.selfattention import SABlock
from networks.mySAblock import SABlock


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        stdvar: float = 1.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, stdvar)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        #x = x + self.attn(self.norm1(x))
        x_, nll = self.attn(self.norm1(x))
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, nll

