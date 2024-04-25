import torch

def kl_divergence(p, q):
    # 确保p和q的所有元素都大于0，以避免计算log(0)
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    
    # 计算KL散度
    kl_div = torch.sum(p * torch.log(p / q))
    return kl_div

# 示例概率分布
p = torch.tensor([0.4, 0.1, 0.5])
q = torch.tensor([0.3, 0.3, 0.4])

# 计算KL散度
kl_result = kl_divergence(p, q)
print("KL divergence:", kl_result.item())
