import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int, # 4
        multiple_of: int, # 256 make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        # x乘以两个不同的权重矩阵，其中一个过silu，两者均上采样，再乘以w2压缩回原先的维度
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class FFNSimple(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))