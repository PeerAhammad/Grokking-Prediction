import torch
import torch.nn as nn

from config import (
    VOCAB_SIZE,
    MODULUS,
    EMBED_DIM,
    NUM_HEADS,
    MLP_DIM,
    MAX_SEQUENCE_LENGTH
)


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, MLP_DIM),
            nn.GELU(),
            nn.Linear(MLP_DIM, EMBED_DIM)
        )

    def forward(self, x):

        normalized = self.norm1(x)

        mask = torch.triu(
            torch.ones(
                x.size(1),
                x.size(1),
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False
        )

        x = x + attention_output
        x = x + self.mlp(self.norm2(x))

        return x


class GrokkingTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM
        )

        self.position_embedding = nn.Embedding(
            MAX_SEQUENCE_LENGTH,
            EMBED_DIM
        )

        self.block = TransformerBlock()

        self.final_norm = nn.LayerNorm(EMBED_DIM)

        self.output_layer = nn.Linear(
            EMBED_DIM,
            MODULUS
        )

    def forward(self, x):

        batch_size, sequence_length = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        x = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        x = self.block(x)

        x = self.final_norm(x)

        # Predict the answer using the final token
        x = x[:, -1, :]

        return self.output_layer(x)