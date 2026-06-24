"""Full encoder-decoder model for handwritten math recognition: forward pass and generate. (Pranav)"""

import torch
import torch.nn as nn

from ml.decoder import TransformerDecoder
from ml.encoder import CNNEncoder


class Im2LatexModel(nn.Module):
    """Combines an image encoder with a Transformer decoder to produce LaTeX token sequences."""

    def __init__(self, vocab_size: int, embed_dim: int = 512) -> None:
        """Initialize the encoder and decoder submodules."""
        super().__init__()
        raise NotImplementedError()

    def forward(self, images: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Compute training-time logits given a batch of images and teacher-forced target tokens."""
        raise NotImplementedError()

    def generate(self, images: torch.Tensor, max_len: int = 150) -> torch.Tensor:
        """Autoregressively decode LaTeX token ids for a batch of images at inference time."""
        raise NotImplementedError()
