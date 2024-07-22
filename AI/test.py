import numpy
import torch
import torch.nn as nn

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
        self.gate_layer = nn.Linear(in_features, out_features)
        self.value_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_gate = self.swish(self.gate_layer(x))
        x_val = self.value_layer(x)
        return x_gate * x_val
    

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, pre_norm=True):
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
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

class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads, pre_norm=True):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)
        self.pre_norm = pre_norm
        self.norm = RMSNorm(dim)

    def create_mask(self, size):
        mask = torch.ones(size, size).triu(diagonal=1)
        return mask
    
    def scale_dot_attention(self, q, k, v, mask):
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=q.device))
        logits = torch.matmul(q, k.transpose(-1, -2)) / scale
        if mask:
            logits += mask * float('-inf')
        attn_weight = torch.softmax(logits, dim=-1)
        return torch.matmul(attn_weight, v)
        


    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        if mask == None:
            mask = self.create_mask(seq_len)
        ori_x = x
        if self.pre_norm:
            x = self.norm(x)
        Q = self.wq(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_score = self.scale_dot_attention(Q, K, V, mask)
        attn_score = attn_score.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out = self.fc_out(attn_score)
        if self.pre_norm:
            return out + ori_x
        else:
            return self.norm(out + x)

    