import torch
import torch.nn as nn
from abc import ABC
from .simple_rl_loss import PairWiseLoss


class RewardModelTrainer(ABC):
    def __init__(self, model, tokenizer, train_dataset, use_margin=True):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.use_margin = use_margin
        self.loss_fn = PairWiseLoss()

    def concat_input(self, chosen_ids, chosen_mask, reject_ids, reject_mask):
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape(-1), reject_ids.shape(-1))
        input_ids = torch.cat((pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id), pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id)), dim=0)
        masks = torch.car((pad_to_length(chosen_mask, max_length, 0), pad_to_length(reject_mask, max_length, 0)), dim=0)
        return input_ids, masks


    def concat_forward(self, chosen_ids, chosen_mask, reject_ids, reject_mask):
        input_ids, all_mask = self.concat_input(chosen_ids, chosen_mask, reject_ids, reject_mask)
        all_values = self.model(input_ids, attention_mask=all_mask)
        chosen_rewards = all_values[:chosen_ids.shape(0)]
        reject_rewards = all_values[chosen_ids.shape(0):]
        return chosen_rewards, reject_rewards



    def train(self):
        self.model.train()
        for chosen, reject, margin in self.train_dataset:
            chosen_tokens = self.tokenizer(chosen)
            chosen_ids = chosen_tokens['input_ids']
            chosen_mask = chosen_tokens['attention_mask']
            reject_tokens = self.tokenizer(reject)
            reject_ids = reject_tokens['input_ids']
            reject_mask = reject_tokens['attention_mask']
            if self.use_margin:
                margin = torch.tensor(margin).to(torch.cuda.current_device())
            else:
                margin = None
            chosen_rewards, reject_rewards = self.concat_forward(chosen_ids, chosen_mask, reject_ids, reject_mask)

            loss = self.loss_fn(chosen_rewards, reject_rewards, margin)
            self.model.backward(loss)
