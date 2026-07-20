"""Loads TrOCR (or Tesseract as a fallback) and runs text recognition on handwritten text blocks. (Pranav)"""

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TextOcr:
    """Wraps a TrOCR model for recognizing handwritten text blocks."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load the TrOCR model and processor from the given model name or path."""
        self.device = torch.device(device)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        """Run inference on a cropped text block image, returning (text, confidence)."""
        pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            output = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

        text = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        return text, self._sequence_confidence(output)

    def _sequence_confidence(self, output) -> float:
        """Mean max-softmax probability across generated steps, as a simple confidence proxy."""
        if not output.scores:
            return 0.0
        step_probs = [torch.softmax(step_logits, dim=-1).max(dim=-1).values for step_logits in output.scores]
        return torch.stack(step_probs).mean().item()
