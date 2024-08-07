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

    def forward(self, log_probs, old_log_probs, advantages, action_mask):
        ratio = (log_probs - old_log_probs).exp()
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
    
    def forward(self, values, old_values, returns, action_mask):
        if self.clip_bound is not None:
            values_clip = old_values + (values - old_values).clamp(-self.clip_bound, self.clip_bound)
            surr1 = (values - returns) ** 2
            surr2 = (values_clip - returns) ** 2
            # max 如果surr1较大，由于其已经限制了更新幅度，因此取它也是安全的，这样可以避免每次更新都过小
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
    
class DPOLoss(nn.Module):
    def __init__(self, loss_type='dpo', beta=0.1):
        super.__init__()
        self.beta = beta
        self.loss_type = loss_type

    def forward(self,
                policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor,
                ref_chosen_logps: torch.FloatTensor,
                ref_rejected_logps: torch.FloatTensor,
                chosen_position_kl: torch.FloatTensor,
                rejected_position_kl: torch.FloatTensor):
        policy_logp_ratio = policy_chosen_logps - policy_rejected_logps
        ref_logp_ratio = ref_chosen_logps - ref_rejected_logps
        if self.loss_type == 'dpo':
            loss = -F.logsigmoid(self.beta * (policy_logp_ratio - ref_logp_ratio))
        elif self.loss_type == 'raft':
            loss = -policy_chosen_logps
        elif self.loss_type == 'tdpo':
            tdpo_alpha = 0.5
            logits = policy_logp_ratio - ref_logp_ratio - tdpo_alpha * (rejected_position_kl - chosen_position_kl.detach())
            loss = -F.logsigmoid(self.beta * logits)
        return loss.mean()