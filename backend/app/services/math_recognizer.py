"""Loads the trained math encoder-decoder model and runs inference on a single math block image. (Pranav)"""

import torch
from PIL import Image
from torchvision import transforms

from ml.image_utils import letterbox
from ml.model import Im2LatexModel
from ml.tokenizer import LatexTokenizer

_to_tensor = transforms.ToTensor()


class MathRecognizer:
    """Wraps the encoder-decoder model defined in ml/model.py for serving."""

    def __init__(
        self,
        weights_path: str,
        vocab_path: str,
        device: str = "cpu",
        embed_dim: int = 512,
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        """Load model weights onto the given device."""
        self.device = torch.device(device)
        self.image_size = image_size
        self.tokenizer = LatexTokenizer(vocab_path=vocab_path)

        self.model = Im2LatexModel(vocab_size=self.tokenizer.vocab_size, embed_dim=embed_dim)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Run inference on a cropped math block image, returning (latex_string, confidence)."""
        # Must match training preprocessing exactly (scripts/preprocess_data.py) — the
        # model was only ever shown images at this fixed size.
        resized = letterbox(image, self.image_size)
        image_tensor = _to_tensor(resized).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        with torch.no_grad():
            token_ids = self.model.generate(image_tensor).squeeze(0)
            latex = self.tokenizer.decode(token_ids.tolist())
            confidence = self._sequence_confidence(image_tensor, token_ids)

        return latex, confidence

    def _sequence_confidence(self, image_tensor: torch.Tensor, token_ids: torch.Tensor) -> float:
        """Mean softmax probability the model assigns to its own greedily-decoded tokens.

        generate() doesn't retain per-step logits, so this re-scores the emitted sequence
        with a single teacher-forced forward pass as a cheap confidence proxy.
        """
        if token_ids.size(0) < 2:
            return 0.0

        input_tokens, target_labels = token_ids[:-1].unsqueeze(0), token_ids[1:]
        logits = self.model(image_tensor, input_tokens)
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[0, torch.arange(logits.size(1)), target_labels]
        return token_probs.mean().item()
