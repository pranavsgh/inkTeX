"""Transformer decoder that autoregressively generates LaTeX tokens from encoder features. (Pranav)"""

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """Autoregressive Transformer decoder over the LaTeX token vocabulary."""

    def __init__(self, vocab_size: int, embed_dim: int = 512, num_layers: int = 6, num_heads: int = 8) -> None:
        """Initialize token embeddings, positional encoding, and decoder transformer blocks."""
        super().__init__()
        raise NotImplementedError()

    def forward(self, encoder_features: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits (B, T, vocab_size) given encoder features and target token ids."""
        raise NotImplementedError()
