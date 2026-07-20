"""Transformer decoder that autoregressively generates LaTeX tokens from encoder features. (Pranav)"""

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """Autoregressive Transformer decoder over the LaTeX token vocabulary."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_len: int = 512,
    ) -> None:
        """Initialize token embeddings, positional encoding, and decoder transformer blocks."""
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_features: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits (B, T, vocab_size) given encoder features and target token ids."""
        seq_len = target_tokens.size(1)
        x = self.token_embedding(target_tokens) + self.pos_embedding[:, :seq_len, :]
        # Right-padding convention (see ml/dataset.py) means every token only ever attends
        # to real tokens before it, so no key-padding mask is needed alongside this causal mask.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(target_tokens.device)
        decoded = self.transformer(tgt=x, memory=encoder_features, tgt_mask=causal_mask)
        return self.output_head(decoded)
