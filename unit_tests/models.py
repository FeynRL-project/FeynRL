import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim*2, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        x = self.embed(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(logits)

class TinyValueModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim*2, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        x = self.embed(input_ids)
        x = self.transformer(x)
        values = self.value_head(x) # [B, T, 1]
        
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(values)
