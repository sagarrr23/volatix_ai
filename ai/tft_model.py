import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.ff(x)))


class GatedResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.linear(x) * self.sigmoid(self.gate(x))


class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_size=64,
                 seq_len=20,
                 heads=4,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Encoder stack
        self.encoder = nn.ModuleList([
            nn.Sequential(
                GatedResidualBlock(hidden_size),
                SelfAttentionBlock(hidden_size, heads, dropout),
                GatedResidualBlock(hidden_size),
                FeedForwardBlock(hidden_size, dropout=dropout),
            ) for _ in range(num_layers)
        ])

        # Decoder (flatten encoder output)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * seq_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 classes: Sell, Wait, Buy
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)

        for block in self.encoder:
            x = block(x)

        # Flatten encoder output for classification heads
        x = x.reshape(x.size(0), -1)  # (batch, seq_len * hidden)

        h = self.decoder(x)

        return self.direction_head(h), self.confidence_head(h), self.reward_head(h)
