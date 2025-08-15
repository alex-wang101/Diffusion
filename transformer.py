import torch
import torch.nn as nn
import torch.nn.functional as f
from model import Config


class AttentionHead(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        # Initializes the key value and query matricies
        # n_embed - input dimension head_size - output dimension
        self.query = nn.Linear(config.n_embed, head_size)
        self.key = nn.Linear(config.n_embed, head_size)
        self.value = nn.Linear(config.n_embed, head_size)
        # Attention masking
        tril = torch.tril(torch.ones(config.cw_size))
        self.register_buffer("tril", tril)
        self.dropout = nn.Dropout(p=Config.p_dropout)
