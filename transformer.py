import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def forward(self, x):
        # Batch, context window, n_embed (shape of input tensor)
        b, t, d = x.shape
        q = self.query[x] # what info am i looking for? 
        k = self.key[x] # What information corresponds?
        v = self.value[x] # What information do i want to communicate during aggregation of token values
        
        att = q @ k.transpose(1, 2) * (d ** -0.5)
        att = att.masked_filled(self.tril == 0, float('-inf'))
        F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v 
        return out
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):


