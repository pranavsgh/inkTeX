"""CNN or ViT encoder that extracts visual features from handwritten math images. (Pranav)"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Convolutional backbone that maps an input image to a sequence of feature vectors."""

    def __init__(self, embed_dim: int = 512, max_seq_len: int = 1024) -> None:
        """Initialize convolutional layers and the projection to embed_dim."""
        # max_seq_len must cover (image_size / 8)**2 (three MaxPool2d(2) layers below) —
        # 1024 matches preprocess_data.py's default 256x256 training image size.
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, embed_dim, 3, padding=1), nn.ReLU(),
        )
        # Learned positional embedding over flattened spatial positions, sliced to the
        # actual feature-map size at forward time (inputs are preprocessed to a fixed size).
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of images (B, C, H, W) to a sequence of features (B, S, embed_dim)."""
        features = self.conv_stack(images)  # (B, embed_dim, H', W')
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        seq_len = features.size(1)
        return features + self.pos_embedding[:, :seq_len, :]


class ViTEncoder(nn.Module):
    """Vision Transformer backbone that maps an input image to a sequence of patch embeddings."""

    def __init__(
        self,
        embed_dim: int = 512,
        patch_size: int = 16,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 512,
    ) -> None:
        """Initialize patch embedding, positional encoding, and transformer encoder blocks."""
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of images (B, C, H, W) to a sequence of patch embeddings (B, S, embed_dim)."""
        patches = self.patch_embed(images)  # (B, embed_dim, H/P, W/P)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        seq_len = patches.size(1)
        patches = patches + self.pos_embedding[:, :seq_len, :]
        return self.transformer(patches)
