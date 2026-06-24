"""Loads the trained math encoder-decoder model and runs inference on a single math block image. (Pranav)"""

from PIL import Image


class MathRecognizer:
    """Wraps the encoder-decoder model defined in ml/model.py for serving."""

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        """Load model weights onto the given device."""
        raise NotImplementedError()

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Run inference on a cropped math block image, returning (latex_string, confidence)."""
        raise NotImplementedError()
