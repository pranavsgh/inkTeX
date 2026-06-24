"""Loads TrOCR (or Tesseract as a fallback) and runs text recognition on handwritten text blocks. (Mutha)"""

from PIL import Image


class TextOcr:
    """Wraps a TrOCR model for recognizing handwritten text blocks."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load the TrOCR model and processor from the given model name or path."""
        raise NotImplementedError()

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        """Run inference on a cropped text block image, returning (text, confidence)."""
        raise NotImplementedError()
