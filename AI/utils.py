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


class Swish(nn.Module):
    """ Swish Activation Function """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class SwiGLU(nn.Module):
    """ SwiGLU Activation Function """
    def __init__(self, input_features, output_features):
        super().__init__()
        # 定义两组权重和偏置，用于两个不同的线性变换
        self.fc_gate = nn.Linear(input_features, output_features)
        self.fc_value = nn.Linear(input_features, output_features)
        self.swish = Swish()  # 可以设置beta值，这里使用默认的beta=1.0

    def forward(self, x):
        # 分别计算门控信号和值信号
        gate = self.swish(self.fc_gate(x))
        value = self.fc_value(x)
        # 进行门控，即元素级别相乘
        return gate * value