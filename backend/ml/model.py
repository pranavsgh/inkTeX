"""Full encoder-decoder model for handwritten math recognition: forward pass and generate. (Pranav)"""

import torch
import torch.nn as nn

from ml.decoder import TransformerDecoder
from ml.encoder import CNNEncoder
from ml.tokenizer import EOS_TOKEN, SOS_TOKEN, SPECIAL_TOKENS

SOS_ID = SPECIAL_TOKENS.index(SOS_TOKEN)
EOS_ID = SPECIAL_TOKENS.index(EOS_TOKEN)


class Im2LatexModel(nn.Module):
    """Combines an image encoder with a Transformer decoder to produce LaTeX token sequences."""

    def __init__(self, vocab_size: int, embed_dim: int = 512) -> None:
        """Initialize the encoder and decoder submodules."""
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.decoder = TransformerDecoder(vocab_size=vocab_size, embed_dim=embed_dim)

    def forward(self, images: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Compute training-time logits given a batch of images and teacher-forced target tokens."""
        encoder_features = self.encoder(images)
        return self.decoder(encoder_features, target_tokens)

    def generate(self, images: torch.Tensor, max_len: int = 150) -> torch.Tensor:
        """Autoregressively decode LaTeX token ids for a batch of images at inference time."""
        was_training = self.training
        self.eval()
        device = images.device
        batch_size = images.size(0)

        tokens = torch.full((batch_size, 1), SOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            encoder_features = self.encoder(images)
            for _ in range(max_len - 1):
                logits = self.decoder(encoder_features, tokens)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == EOS_ID)
                if finished.all():
                    break

        self.train(was_training)
        return tokens
