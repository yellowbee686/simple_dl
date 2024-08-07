import numpy
import torch
import torch.nn as nn
from types import Tuple


class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_size))
        self.beta = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        return self.scale * (x-x_mean) / (x_std + self.eps) + self.beta
    
class RMSNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms
    
class BatchNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-8, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.feature_size = feature_size
        self.scale = nn.Parameter(torch.ones(1, feature_size, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, feature_size, 1, 1))
        self.register_buffer('running_mean', torch.zeros(feature_size))
        self.register_buffer('running_var', torch.ones(feature_size))

    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            batch_var = torch.var(x, dim=[0, 2, 3], keepdim=True)
            x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * batch_var
        else:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            
        out = self.scale * x + self.beta
        return out
    
class Swish(nn.Module):
    def __init__(self, beta=1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, beta=1.0) -> None:
        super().__init__()
        self.swish = Swish(beta)
        self.gate_layer = nn.Linear(in_features, out_features, bias=False)
        self.value_layer = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x_gate = self.swish(self.gate_layer(x))
        x_val = self.value_layer(x)
        return x_gate * x_val
    

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, pre_norm=True):
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.pre_norm = pre_norm
        self.norm = RMSNorm(dim)

    def forward(self, x):
        if self.pre_norm:
            x_norm = self.norm(x)
            x_out = self.w3(self.swiglu(x_norm)) + x
        else:
            x_out = self.w3(self.swiglu(x)) + x

        if not self.pre_norm:
            x_out = self.norm(x_out)
        return x_out


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads, max_batch_size, max_seq_len, freq_cis, num_kv_heads=0):
        if num_kv_heads > 0:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        assert self.n_rep * self.num_kv_heads == self.num_heads, "num_heads must be devisible by num_kv_heads"
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim * self.num_kv_heads, bias=False)
        self.wv = nn.Linear(dim, self.head_dim * self.num_kv_heads, bias=False)
        self.fc_out = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.freq_cis = freq_cis
        self.cache_k = torch.zeros((max_batch_size, max_seq_len, self.num_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, self.num_kv_heads, self.head_dim))

    def create_mask(self, size):
        mask = torch.ones(size, size).triu(diagonal=1)
        return mask
    
    def scale_dot_attention(self, Q, K, V, mask):
        scale = torch.sqrt(torch.tensor(self.head_dim))
        logits = torch.matmul(Q, K.transpose(-1, -2)) / scale
        if mask:
            logits += mask * float('-inf')
        attn_weight = torch.softmax(logits, dim=-1)
        return torch.matmul(attn_weight, V)


    def forward(self, x, start_pos=0, mask=None):
        batch_size, seq_len, _ = x.size()
        if mask == None:
            mask = self.create_mask(seq_len)
        x = self.norm(x)
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        Q, K = apply_rotary_emb(Q, K, self.freq_cis)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = K
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = V
        K = self.cache_k[:batch_size, :start_pos+seq_len]
        V = self.cache_v[:batch_size, :start_pos+seq_len]
        K = K.repeat(1, 1, self.n_rep, 1).transpose(1, 2)
        V = V.repeat(1, 1, self.n_rep, 1).transpose(1, 2)
        Q = Q.transpose(1, 2)
        attn_score = self.scale_dot_attention(Q, K, V, mask).transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out = x + self.fc_out(attn_score)
        return out

    
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_hidden_dim, num_heads, max_batch_size, max_seq_len, freqs_cis, pre_norm=True, num_kv_heads=0):
        super().__init__()
        self.attention = SimpleAttention(dim, num_heads, max_batch_size, max_seq_len, freqs_cis, pre_norm, num_kv_heads)
        self.ffn = FFN(dim, ffn_hidden_dim, pre_norm)

    def forward(self, x, start_pos = 0, mask=None):
        attn_out = self.attention(x, start_pos, mask)
        return self.ffn(attn_out)


class Transformer(nn.Module):
    def __init__(self, num_blocks, vocab_size, dim, ffn_hidden_dim, num_heads, max_batch_size, max_seq_len, pre_norm=True, num_kv_heads=0):
        super().__init__()
        self.blocks = []
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len)
        for i in range(num_blocks):
            self.blocks.append(TransformerBlock(dim, ffn_hidden_dim, num_heads, max_batch_size, max_seq_len, self.freqs_cis, pre_norm, num_kv_heads))
        
        self.norm = RMSNorm(dim)
        self.out_layer = nn.Linear(dim, vocab_size)

    def forward(self, x, start_pos = 0, mask=None):
        for block in self.blocks:
            x = block(x, start_pos, mask)
        x = self.norm(x)
        out = self.out_layer(x)
        return out