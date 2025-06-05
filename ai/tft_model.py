import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_size=128,
                 seq_len=30,
                 heads=4,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Projection and Positional Encoding
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=seq_len)

        # Transformer Encoder
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(hidden_size, heads, dropout)
            for _ in range(num_layers)
        ])

        # Sequence flattening + bottleneck decoder
        self.decoder = nn.Sequential(
            nn.Flatten(),  # (batch, seq_len * hidden_size)
            nn.Linear(hidden_size * seq_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # Output Heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5)  # 🔥 5-class classification
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)                        # (batch, seq_len, hidden)
        x = self.pos_encoding(x)                      # Add time step info
        x = self.encoder(x)                           # Transformer layers
        h = self.decoder(x)                           # Flatten + bottleneck

        # Output predictions
        direction = self.direction_head(h)
        confidence = self.confidence_head(h)
        reward = self.reward_head(h)

        return direction, confidence, reward
