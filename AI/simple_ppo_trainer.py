import torch
import torch.nn as nn
from torch import distributed as dist
from abc import ABC
from .simple_rl_loss import PolicyLoss, ValueLoss, SFTLoss
from .simple_experience import Experience, ExperienceMaker, ReplayBuffer, get_log_probs
from transformers import AdamW, get_scheduler
import math

class PPOTrainer(ABC):
    def __init__(self, actor, ref_model, critic, reward_model, tokenizer, args):
        self.actor = actor
        # 初始和actor相同，但会随着训练逐步copy actor
        self.ref_model = ref_model
        self.critic = critic
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.experience_maker = ExperienceMaker(actor, ref_model, critic, reward_model, tokenizer)
        self.sample_batch_size = args.sample_batch_size
        self.rollout_batch_size = args.rollout_batch_size
        self.micro_rollout_batch_size = args.micro_rollout_batch_size
        self.replay_buffer = ReplayBuffer(self.sample_batch_size) # 取合适的samples做advantage normalize
        self.actor_loss_fn = PolicyLoss(clip_bound=0.2)
        self.critic_loss_fn = ValueLoss(value_loss_ratio=0.5, clip_bound=0.2)
        self.ptx_loss_fn = SFTLoss() # 保持RL不要距离SFT太远，实现时每次训练中随机从另一个SFT数据集中sample batch作为label，用actor(data)作为logits来计算loss
        self.actor_lr = 3e-6
        self.critic_lr = 3e-8
        self.max_steps = 100000
        self.actor_optimizer = AdamW(self.actor.paramaters(), lr=self.actor_lr)
        self.actor_scheduler = get_scheduler(
            'cosine',
            self.actor_optimizer,
            num_warmup_steps=math.ceil(self.max_steps * 0.03),
            num_training_steps=self.max_steps,
        )
        self.critic_optimizer = AdamW(self.critic.paramaters(), lr=self.critic_lr)
        self.critic_scheduler = get_scheduler(
            'cosine',
            self.critic_optimizer,
            num_warmup_steps=math.ceil(self.max_steps * 0.03),
            num_training_steps=self.max_steps,
        )



    def update_ref_model(self):
        self.ref_model.load_state_dict(self.actor.state_dict())


    def train(self, train_dataset):
        world_size = dist.get_world_size()
        update_steps = self.rollout_batch_size // (world_size * self.micro_rollout_batch_size)
        steps = 1
        for prompts in train_dataset:
            experience = self.experience_maker.make_experience(prompts)
            self.replay_buffer.append(experience)
            if steps % update_steps == 0:
                self.replay_buffer.normalize()

    
    def ppo_train(self):
        train_steps = self.rollout_batch_size // self.sample_batch_size
        self.actor.train()
        self.critic.train()
        for i in range(train_steps):
            train_experience = self.replay_buffer.sample()
            self.actor_step(train_experience, i)
            self.critic_step(train_experience, i)



    def actor_step(self, experience: Experience, global_step):
        response_length = experience.response_log_probs.size(1)
        actor_log_probs = get_log_probs(self.actor, experience.sequences, response_length, experience.response_mask)
        actor_loss = self.actor_loss_fn(actor_log_probs, experience.response_log_probs, experience.advantages, experience.response_mask)
        self.actor.backward(actor_loss)
        self.actor_optimizer.step()
        self.actor_scheduler.step()

    def critic_step(self, experience: Experience, global_step):
        response_length = experience.response_log_probs.size(1)
        values = self.critic(experience.sequences, experience.attention_mask)[:, -response_length:]
        critic_loss = self.critic_loss_fn(values, experience.values, experience.returns, experience.response_mask)
        self.critic.backward(critic_loss)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

