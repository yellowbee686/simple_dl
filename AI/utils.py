import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        # 一个可学习参数，像BN一样用于rescale
        self.scale = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # 计算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 应用归一化
        x_normalized = x / rms
        # 应用缩放参数
        return x_normalized * self.scale

