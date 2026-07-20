"""POST /analyze — accepts a full handwritten page image, returns ordered text/math blocks. (Mutha)"""

import io

from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.models.schemas import AnalyzeResponse

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("", response_model=AnalyzeResponse)
async def analyze_page(request: Request, file: UploadFile) -> AnalyzeResponse:
    """Run layout detection on the uploaded page and return ordered blocks with bboxes and types."""
    layout_detector = request.app.state.layout_detector
    if layout_detector is None:
        raise HTTPException(status_code=503, detail="Layout detection model is not loaded")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    blocks = layout_detector.detect(image)

    # Text content is resolved here via TrOCR. Math content is deliberately left empty —
    # recognizing it means calling Pranav's MathRecognizer, which /convert does when it
    # routes each block to the math or text model; duplicating that here would couple
    # this file to a model it doesn't own.
    text_ocr = request.app.state.text_ocr
    if text_ocr is not None:
        for block in blocks:
            if block.type != "text":
                continue
            crop_box = (int(block.bbox.x1), int(block.bbox.y1), int(block.bbox.x2), int(block.bbox.y2))
            text, confidence = text_ocr.recognize(image.crop(crop_box))
            block.content = text
            block.confidence = confidence

    return AnalyzeResponse(blocks=blocks, image_width=image.width, image_height=image.height)
