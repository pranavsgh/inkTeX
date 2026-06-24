"""Loads the YOLOv8 layout model and runs page segmentation into text/math regions. (Mutha)"""

from PIL import Image

from app.models.schemas import Block


class LayoutDetector:
    """Wraps a YOLOv8 model trained to detect text and math regions on a page."""

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        """Load YOLOv8 weights onto the given device."""
        raise NotImplementedError()

    def detect(self, image: Image.Image) -> list[Block]:
        """Run page segmentation, returning ordered blocks with bounding boxes and type labels."""
        raise NotImplementedError()
