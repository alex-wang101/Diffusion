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
        tril = torch.tril(torch.ones(config.cw_size, config.cw_size))
        self.register_buffer("tril", tril)
        self.dropout = nn.Dropout(p=Config.p_dropout)
    
    def forward(self, x):
        # Batch, context window, n_embed (shape of input tensor)
        b, t, d = x.shape
        q = self.query(x) # what info am i looking for? 
        k = self.key(x) # What information corresponds?
        v = self.value(x) # What information do i want to communicate during aggregation of token values
        
        att = q @ k.transpose(1, 2) * (d ** -0.5)
        att = att.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v 
        return out
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_heads 
        self.attention_heads = nn.ModuleList(
            [AttentionHead(config, head_size) for _ in range(config.n_heads)]
        )
    def forward(self, x):
        x = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        return x 

# normalizes the output of 
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        # The scale parameter
        self.gemma = nn.Parameter(torch.ones(dim))
        # The shift parameter
        self.beta = nn.Parameter(torch.zeros(dim))
        # small constant 1e-5 to avoid division by zero
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_var = x.var(-1, keepdim=True)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        x_norm = self.gemma * x_norm + self.beta
        return x_norm

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(p=Config.p_dropout)
        )

    def forward(self, x):
        return self.net(x)

# Construct the transformer block, putting everything together
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadedAttention(config)
        self.ln1 = LayerNorm(config.n_embed)
        self.ln2 = LayerNorm(config.n_embed)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln2(x) 
        x = x + self.ff(x)
        return x
