import torch
import torch.nn as nn
import torch.nn.functional as F


class SFTLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super.__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[:, 1:].contiguous().vciew(-1)
        return self.loss(shift_logits, shift_labels)
    

class PolicyLoss(nn.Module):
    def __init__(self, clip_bound=0.2):
        super.__init__()
        self.clip_bound = clip_bound

    def forward(self, log_probs, ref_log_probs, advantages, action_mask):
        ratio = (log_probs - ref_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_bound, 1+ self.clip_bound) * advantages
        # -min
        loss = -torch.min(surr1, surr2)
        loss = (loss * action_mask).sum(axis=-1) / action_mask.sum(axis=-1)
        return loss


class ValueLoss(nn.Module):
    def __init__(self, value_loss_ratio=0.5, clip_bound=None):
        super.__init__()
        self.value_loss_ratio = value_loss_ratio
        self.clip_bound = clip_bound
    
    def forward(self, values, ref_values, returns, action_mask):
        if self.clip_bound is not None:
            values_clip = ref_values + (values - ref_values).clamp(-self.clip_bound, self.clip_bound)
            surr1 = (values - returns) ** 2
            surr2 = (values_clip - returns) ** 2
            # max
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = (loss * action_mask).sum(axis=-1) / action_mask.sum(axis=-1)
        return self.value_loss_ratio * loss


class PairWiseLoss(nn.Module):
    def forward(self, chosen_reward, reject_reward, margin = None):
        if margin:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()
    
