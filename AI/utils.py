import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, feature_shape, eps = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_shape))
        self.eps = eps

    def forward(self, x):
        mean = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / mean
    

class LayerNorm(nn.Module):
    def __init__(self, feature_shape, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_shape))
        self.beta = nn.Parameter(torch.zeros(feature_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma*(x - mean) / (std + self.eps) + self.beta
        

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)
    

class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, beta):
        super().__init__()
        self.swish = Swish(beta)
        self.gate_layer = nn.Linear(in_features, out_features)
        self.value_layer = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        gate = self.swish(self.gate_layer(x))
        value = self.value_layer(x)
        return gate * value