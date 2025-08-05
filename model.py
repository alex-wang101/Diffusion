import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
@dataclass
class Config: 
    cw_size : int = 8
    batch_size : int = 4
    train_split : float = 0.9
    n_embd : int = 32

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
        print(data)
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
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, inputs, targets):
        embeded_tokens = self.token_embedding_table(inputs)
        logits = self.lm_head(embeded_tokens)

        # b = batch size
        # t = context window size
        # c = embedding dimension

        b, t, c = logits.shape

        # Flatten the logits to the proper shape for the loss function
        logits = logits.view(b*t, -1)
        targets = targets.view(b*t)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
# reads the input text file, initializes the Data class, and retrieves a batch of training data.
def train_test_model():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    config = Config()
    data = Data(text, config)
    xbatch, ybatch = data.get_batch("train", config)
    # Model instantiation
    model = languageModel(config).to(device)
    model(xbatch, ybatch)
    logits, loss = model(xbatch, ybatch)
    print("Logits shape:", logits.shape)
    print("loss:", loss.item())
def main():
    train_test_model()

if __name__ == "__main__":
    main()
