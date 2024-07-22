import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, feature_shape, eps = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms
    

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
            with torch.no_grad():
                self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1-self.momentum) * batch_var

        x_mean = self.running_mean
        x_var = self.running_var
        
        x = (x - x_mean) / torch.sqrt(x_var + self.eps)
        out = self.scale * x + self.beta
        return out


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


class SimpleSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))    