import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import RMSNorm

class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        self.hidden_dim = hidden_dim
        self.num_heads =num_heads
        self.head_dim = hidden_dim // num_heads
        assert (self.head_dim * num_heads == hidden_dim), "must be divisible"
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _create_look_ahead_mask(self, size):
        mask = torch.ones(size, size).triu(diagonal=1)
        return mask
    
    def _scale_dot_attention(self, q, k, v, mask):
        # q k v均为 [B, n_head, seq, h_dim]
        qk = torch.matmul(q, k.transpose(-1, -2)) # [B, n_head, seq, seq]
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        logits = qk / scale
        if mask:
            logits += mask * float('-inf')
        attention_weights = F.softmax(logits, dim=-1) # [B, n_head, seq, seq]
        output = torch.matmul(attention_weights, v) # [B, n_head, seq, h_dim]
        return output

    def forward(self, x, mask):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        x = self.norm(x)
        if mask == None:
            mask = self._create_look_ahead_mask(sequence_length)
        Q = self.Wq(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        self.attn_score = self._scale_dot_attention(Q, K, V, mask).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.fc_out(self.attention_output) + x
        return output
