import torch
import torch.nn as nn
import torch.nn.functional as F


class PairWiseLoss(nn.Module):
    def forward(self, chosen_reward, reject_reward, margin = None):
        if margin:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()
    
