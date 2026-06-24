"""CNN or ViT encoder that extracts visual features from handwritten math images. (Pranav)"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Convolutional backbone that maps an input image to a sequence of feature vectors."""

    def __init__(self, embed_dim: int = 512) -> None:
        """Initialize convolutional layers and the projection to embed_dim."""
        super().__init__()
        raise NotImplementedError()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of images (B, C, H, W) to a sequence of features (B, S, embed_dim)."""
        raise NotImplementedError()


class ViTEncoder(nn.Module):
    """Vision Transformer backbone that maps an input image to a sequence of patch embeddings."""

    def __init__(self, embed_dim: int = 512, patch_size: int = 16) -> None:
        """Initialize patch embedding, positional encoding, and transformer encoder blocks."""
        super().__init__()
        raise NotImplementedError()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of images (B, C, H, W) to a sequence of patch embeddings (B, S, embed_dim)."""
        raise NotImplementedError()
