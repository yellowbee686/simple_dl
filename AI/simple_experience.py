import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from dataclasses import dataclass
from abc import ABC


def get_log_probs(model, sequences, response_length, response_mask):
    output = model(sequences, attention_mask=response_mask)
    log_probs = label_log_probs(output['logits'][:, :-1, :], sequences[:, 1:])
    return log_probs[: -response_length:]


# 从[B, S, V]的logits中提取[B, S]的log_probs 每一步只取生成的token_id对应的log_probs
def label_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    response_mask: (B, A)
    S is prompt + response length
    A is response length
    """
    sequences: torch.Tensor
    response_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor


class ExperienceMaker(ABC):
    def __init__(self, actor, ref_model, critic, reward_model, tokenizer):
        self.actor = actor
        self.ref_model = ref_model
        self.critic = critic
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.prompt_max_len = 8192
        self.max_len = self.prompt_max_len + 2048
        self.kl_coef = 0.02
        self.gamma = 1 # reward是否衰减 在LLM中设为1比较合理，目前的设置不符合越远的贡献越低的问题
        self.lambd = 0.95 # TD-learning向后看多少步，1时variance最大 bias最小，0时为TD(0) variance最小 bias最大

    def compute_reward(self, reward, log_probs, ref_log_probs, response_mask):
        # approx kl
        kl = (log_probs - ref_log_probs) * response_mask
        kl_reward = -self.kl_coef * kl
        reward = reward.clamp(min=-10, max=10)
        # reward assign to last
        eos_indices = response_mask.size(1) - 1 - response_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=reward.unsqueeze(1).to(kl.dtype))
        return last_reward + kl_reward, kl
    
    # GAE
    def compute_advantage(self, values, rewards, response_mask):
        values = values * response_mask
        rewards = rewards * response_mask
        response_length = rewards.size(1)
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages, returns



    def make_experience(self, prompts):
        self.actor.eval()
        self.critic.eval()
        self.ref_model.eval()
        self.reward_model.eval()

        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt",
            max_length=self.prompt_max_len,
            padding=True,
            truncation=True)
        input_ids = inputs['input_ids']
        input_length = input_ids.size(1)
        input_attention_mask = inputs['attention_mask']
        # do_sample为False时是greedy decode 否则top_p top_k仍起作用
        generate_kwargs = {
            'input_ids': input_ids, 
            'attention_mask': input_attention_mask, # prompt padding mask
            'temperature': 1.0,
            'output_scores': True,
            'top_p': 1.0,
            'top_k': -1,
            'do_sample': True,
            'max_length': self.max_len,
            'return_dict_in_generate': True,
        }
        output = self.actor.generate(**generate_kwargs)
        sequences = output.sequences
        # response_length of list of [batch_size, vocab_size] -> tensor [batch_size, response_length, vocab_size]
        logits = output.scores.stack(dim=1)
        responses = sequences[:, input_length:]
        log_probs = label_log_probs(logits, responses)

        # input_ids: [101, 102, 103, 104, 0, 0]  # input_length = 4 + 2 padding
        # response: [201, 202, 50256, 0, 0]      # response_length = 2 + 2 padding
        # sequences: [101, 102, 103, 104, 0, 0, 201, 202, 50256, 0, 0]  # 总长度 = 10
        response_length = sequences.size(1) - input_length
        full_attention_mask = torch.where(responses == self.tokenizer.pad_token_id, 0, 1)
        response_mask = full_attention_mask[: -response_length:]
        ref_log_probs = get_log_probs(self.ref_model, sequences, response_length, full_attention_mask)
        # [batch_size]
        rewards = self.reward_model(sequences, full_attention_mask)
        # [batch_size, response_length]
        values = self.critic(sequences, full_attention_mask)[:, -response_length:]
        rewards, kl = self.compute_reward(rewards, log_probs, ref_log_probs, response_mask)
        advantages, returns = self.compute_advantage(values, rewards, response_mask)
        experience = Experience(
            sequences,
            log_probs,
            ref_log_probs,
            values,
            returns,
            advantages,
            full_attention_mask,
            response_mask,
        )
        return experience



class ReplayBuffer(ABC):
    def __init__(self, sample_batch_size, capacity = 128):
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.capacity = capacity
        self.items = []


    def clear(self):
        self.items.clear()

    def append(self, experience):
        # 不再拆分experience内的一个batch，这里也是有随机性的
        self.items.append(experience)
        if len(self.items) > self.capacity:
            self.items = self.items[(len(self.items) - self.capacity):]

    def sample(self):
        if len(self.items) == 0:
            return None
        item_size = self.items[0].size(0)
        sample_size = self.sample_batch_size // item_size
        assert (sample_size * item_size == self.sample_batch_size), f"invalid sample_batch_size"
        items = random.sample(self.items, sample_size)
        sequences = torch.cat([i.sequences for i in items])
        response_log_probs = torch.cat([i.response_log_probs for i in items])
        values = torch.cat([i.values for i in items])
        returns = torch.cat([i.returns for i in items])
        advantages = torch.cat([i.advantages for i in items])
        attention_mask = torch.cat([i.attention_mask for i in items])
        response_mask = torch.cat([i.response_mask for i in items])
        return Experience(sequences, response_log_probs, values, returns, advantages, attention_mask, response_mask)
    
    def normalize(self):
        # flatten展平成1维
        advs = torch.cat(torch.cat([i.advantages for i in self.items])).flatten()
        response_mask = torch.cat([i.response_mask for i in self.items]).flatten()
        adv_sum = advs.sum()
        cnt = response_mask.sum()
        adv_mean = adv_sum / cnt
        std = ((advs - adv_mean) ** 2 * response_mask).sum()
        # rsqrt为平方根的倒数
        rstd = (std / cnt).clamp(1e-8).rsqrt()
        for i in self.items:
            i.advantages = (i.advantages - adv_mean) * rstd





