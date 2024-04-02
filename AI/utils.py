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


def swiglu(input):
    """
    实现SwiGLU激活函数。
    
    参数:
    - input: 输入张量，假设其最后一个维度是2倍的隐藏维度，
             其中一半用于线性变换，另一半用于门控。
             
    返回:
    - output: 经过SwiGLU激活函数处理后的输出张量。
    """
    # 将输入张量分为两部分，B是输入数据的维度，D是特征维度
    B, D = input.shape[:-1], input.shape[-1]
    x, gate = input.chunk(2, dim=-1)
    
    # 应用门控，这里使用sigmoid函数作为门控机制
    gate = torch.sigmoid(gate)
    
    # 将门控后的部分与另一半相乘
    return x * gate