import torch
import torch.nn as nn
from abc import ABC

class PPOTrainer(ABC):
    def __init__(self, actor, critic, reward_model, tokenizer):
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.tokenizer = tokenizer

    def train(self):
        self.actor