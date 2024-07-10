import torch
import torch.nn as nn
from torch import distributed as dist
from abc import ABC
from .simple_rl_loss import PolicyLoss, ValueLoss
from .simple_experience import ExperienceMaker

class PPOTrainer(ABC):
    def __init__(self, actor, ref_model, critic, reward_model, tokenizer):
        self.actor = actor
        # 初始和actor相同，但会随着训练逐步copy actor
        self.ref_model = ref_model
        self.critic = critic
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.experience_maker = ExperienceMaker(actor, ref_model, critic, reward_model, tokenizer)

    def update_ref_model(self):
        self.ref_model.load_state_dict(self.actor.state_dict())


    def train(self, train_dataset, args):
        world_size = dist.get_world_size()
        update_steps = args.rollout_batch_size // (world_size * args.micro_rollout_batch_size)
        steps = 1
        for prompts in train_dataset:

        self.actor