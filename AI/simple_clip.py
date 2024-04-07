import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class SimpleCLIP(nn.Module):
    def __init__(self, dim: int, dim_i, dim_t, init_logit_scale: float = np.log(1 / 0.07)):
        self.text_encoder = BERTEncoder()
        self.image_encoder = ConvNextEncoder()
        self.text_proj = nn.Linear(dim_t, dim, bias=False)
        self.image_proj = nn.Linear(dim_i, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones(1) * init_logit_scale)

    
    def get_loss(self, images, texts, temperature):
        # images: [B, h, w, c]
        # text: [B, seq]
        batch_size = images.shape[0]
        image_feature = self.image_encoder(images) # [B, dim_i]
        text_feature = self.text_encoder(texts) # [B, dim_t]
        image_embedding = self.image_proj(image_feature).norm(dim=1) # [B, dim]
        text_embedding = self.text_proj(text_feature).norm(dim=1) # [B, dim]
        # 每一行代表一个image和所有text计算logits 每一列代表一个text和所有image计算logits
        image_logits = torch.matmul(image_embedding, text_embedding.T) * self.logit_scale.exp()  # [B, B]
        text_logits = image_logits.T
        labels = torch.arange(batch_size).to(images.device)
        # 首先对每一行求softmax得到prob，然后只取label对应位置的取-log(prob) 再对这些求平均得到loss并minimize
        loss_i = F.cross_entropy(image_logits, labels)
        loss_t = F.cross_entropy(text_logits, labels)
        return (loss_i+loss_t)/2



