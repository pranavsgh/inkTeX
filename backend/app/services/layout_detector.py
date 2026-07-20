"""Loads the YOLOv8 layout model and runs page segmentation into text/math regions. (Mutha)"""

from PIL import Image
from ultralytics import YOLO

from app.models.schemas import Block, BoundingBox

CLASS_TO_TYPE = {0: "text", 1: "math"}


class LayoutDetector:
    """Wraps a YOLOv8 model trained to detect text and math regions on a page."""

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        """Load YOLOv8 weights onto the given device."""
        self.model = YOLO(weights_path)
        self.device = device

    def detect(self, image: Image.Image) -> list[Block]:
        """Run page segmentation, returning ordered blocks with bounding boxes and type labels."""
        results = self.model.predict(image, device=self.device, verbose=False)[0]

        detections = []
        for box in results.boxes:
            block_type = CLASS_TO_TYPE.get(int(box.cls.item()))
            if block_type is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((block_type, (x1, y1, x2, y2), float(box.conf.item())))

        # Approximate reading order: top-to-bottom, then left-to-right.
        detections.sort(key=lambda d: (d[1][1], d[1][0]))

        return [
            Block(
                type=block_type,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                content="",  # filled in downstream by TextOcr or MathRecognizer, not here
                confidence=confidence,
                order=order,
            )
            for order, (block_type, (x1, y1, x2, y2), confidence) in enumerate(detections)
        ]
