import torch

def softmax(z):
    # 从 z 中减去最大值以增加数值稳定性
    z_max = torch.max(z)
    exps = torch.exp(z - z_max)
    sum_exps = torch.sum(exps)
    return exps / sum_exps

# 测试函数
z = torch.tensor([2.0, 1.0, 0.1])
print("Softmax output:", softmax(z))
