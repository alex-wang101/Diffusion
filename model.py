import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import transformer as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
@dataclass
class Config: 
    cw_size : int = 8
    batch_size : int = 4
    train_split : float = 0.9
    n_embed : int = 32
    train_iter : int = 2000
    eval_iter : int = 200
    lr : float = 1e-3
    n_heads : int = 4
    p_dropout : float = 0.2
    
class Data:
    """
    Data class to handle text data, encoding and decoding characters.
    This class also prepares batches of data for training.
    Splits the data into training and testing sets based on the specified configuration.
    Attributes:
        text (str): The input text data.
        config (Config): Configuration object containing parameters like vocabulary size and batch size.
        stend (dict): Mapping from characters to indices.
        stded (dict): Mapping from indices to characters.
        train_data (torch.Tensor): Tensor containing training data.
        test_data (torch.Tensor): Tensor containing testing data.
    """
    def __init__(self, text, config):
        vocab = list(set(text))
        config.vocab = vocab
        print(len(vocab), vocab)
        config.vocab_size = len(vocab)
        self.stend = {ch : i for i, ch in enumerate(vocab)}
        self.stded = {i : ch for i, ch in enumerate(vocab)}

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(config.train_split * len(data))
        self.train_data = data[:n]
        self.test_data = data[n:]

    def encode(self, text):
        return [self.stend[ch] for ch in text]
    def decode(self, indicies):
        return (self.stded[idx] for idx in indicies)
    
    # Generates a batch of data
    def get_batch(self, split, config):
        if split == "train": 
            data = self.train_data 
        elif split == "test":
            data = self.test_data
        index = torch.randint(len(data) - config.cw_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.cw_size] for i in index]).to(device)
        y = torch.stack([data[i+1:i+1+config.cw_size] for i in index]).to(device)
        return x, y

    def estimate_loss(self, model, config):
        pass

class languageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        print(self.lm_head)

    def forward(self, inputs, targets=None):
        """ Forward pass of the model.
        Args:
            inputs (torch.Tensor): Input token indices.
            targets (torch.Tensor): Target token indices for loss computation.
        Returns:
            logits (torch.Tensor): Logits for the next token prediction.
            loss (torch.Tensor): Computed loss value."""
        embeded_tokens = self.token_embedding_table(inputs)
        logits = self.lm_head(embeded_tokens)

        # b = batch size
        # t = context window size
        # c = embedding dimension
        b, t, c = logits.shape

        if targets is not None: 
            # Flatten the logits to the proper shape for the loss function
            logits = logits.view(b*t, -1)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens=1000):
        """
        Generate new tokens based on the input index.
        Args:
            idx (torch.Tensor): Input indices to start generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
        Returns:
            torch.Tensor: Generated token indices.
        """
        for _ in range(max_new_tokens):
            idx_context = idx[:, -Config.cw_size:]  # Use the last 8 tokens as context
            logits, _ = self(idx_context, None) 
            logits = logits[:, -1, :]  # Get the logits for the last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the probability distribution
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

            
       
# reads the input text file, initializes the Data class, and retrieves a batch of training data.
def train_test_model():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    config = Config()
    data = Data(text, config)
    xbatch, ybatch = data.get_batch("train", config)

    # Model instantiation

    # Training loop 
    model = languageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    count = 0
    for i in range(config.train_iter):
        optimizer.zero_grad()
        xbatch, ybatch = data.get_batch("train", config)
        logits, loss = model(xbatch, ybatch)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:    
            print(f'loss: {loss.item()} {count}')
            count += 1
        # print("Logits:", logits)
        # print("loss:", loss)

    # Generate an output tensor
    in_tensor = torch.zeros((1, 1), dtype=torch.long)
    out_tensor = model.generate(in_tensor, max_new_tokens=1000)
    # print(out_tensor)

def test_modules(config: Config):
    b, t, d = 4, 8, 32
    head_size = d // config.n_heads
    module = tf.AttentionHead(config, head_size)
    in_tensor = torch.zeros((b, t, 32))
    out = module(in_tensor)
    out_shape = (4, 8, 8)
    assert out.shape == out_shape, f"failed with shape {out_shape}"

    head_size = d // config.n_heads
    module = tf.MultiHeadedAttention(config)
    in_tensor = torch.zeros((b, t, d))
    out = module(in_tensor)
    out_shape = (4, 8, 32)
    assert out.shape == out_shape, f"failed with shape {out_shape}"

def main():
    config = Config()
    # train_test_model()
    test_modules(config)

if __name__ == "__main__":
    main()
