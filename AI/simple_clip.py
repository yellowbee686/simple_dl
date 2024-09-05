import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class ViTEncoder(nn.Module):
    def __init__(self, vit_model):
        super(ViTEncoder, self).__init__()
        self.vit = vit_model

    def forward(self, images):
        outputs = self.vit(images)
        # Assuming outputs.last_hidden_state contains the token embeddings
        # The first token is the CLS token
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, dim_i]
        return cls_embeddings

class SimpleCLIP(nn.Module):
    def __init__(self, dim: int, dim_i, dim_t, init_logit_scale: float = np.log(1 / 0.07)):
        self.text_encoder = BERTEncoder()
        self.image_encoder = ViTEncoder()
        self.text_proj = nn.Linear(dim_t, dim, bias=False)
        self.image_proj = nn.Linear(dim_i, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones(1) * init_logit_scale)

    
    def get_loss(self, images, texts):
        # images: [B, h, w, c]
        # text: [B, seq]
        batch_size = images.shape[0]
        image_feature = self.image_encoder(images) # [B, dim_i]
        # 使用CLS token的embedding或mean_pooling 最终生成dim_t维度 BERT中使用CLS token
        text_feature = self.text_encoder(texts) # [B, dim_t] 
        # 使用CLS(classification) token的embedding作为整幅图的embedding
        image_embedding = self.image_proj(image_feature) # [B, dim]
        text_embedding = self.text_proj(text_feature) # [B, dim]
        image_embedding = F.norm(image_embedding, dim=1)  # [B, dim]
        text_embedding = F.norm(text_embedding, dim=1)  # [B, dim]
        # 每一行代表一个image和所有text计算logits 每一列代表一个text和所有image计算logits
        # logit_scale 取exp是为了保证这个参数无论正负，logit_scale.exp()始终为正，对logits的影响始终是正常的而不发生错误
        # 学习率控制，这样logit_scale在更新时是log(logit_scale)的速度更新，化乘法为加法，能够使该参数的学习更稳定
        image_logits = torch.matmul(image_embedding, text_embedding.T) * self.logit_scale.exp()  # [B, B]
        text_logits = image_logits.T
        labels = torch.arange(batch_size).to(images.device)
        # 首先对每一行求softmax得到prob，然后只取label对应位置的取-log(prob) 再对这些求平均得到loss并minimize
        loss_i = F.cross_entropy(image_logits, labels)
        loss_t = F.cross_entropy(text_logits, labels)
        return (loss_i+loss_t)/2

     




