import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from .simple_rl_loss import DPOLoss

class DPOTrainer(ABC):
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)