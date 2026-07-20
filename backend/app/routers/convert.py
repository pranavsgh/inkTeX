"""POST /convert — master endpoint: analyze page, route each block, assemble .tex, compile PDF. (Shared)"""

import io
import os
import uuid

from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.models.schemas import ConvertResponse
from app.services.latex_assembler import assemble_latex
from app.services.pdf_compiler import PDFCompilationError, compile_tex_to_pdf

router = APIRouter(prefix="/convert", tags=["convert"])

RESULTS_DIR = "static/results"


@router.post("", response_model=ConvertResponse)
async def convert_page(request: Request, file: UploadFile) -> ConvertResponse:
    """Run the full pipeline on an uploaded page: analyze -> recognize blocks -> assemble -> compile."""
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

    text_ocr = request.app.state.text_ocr
    math_recognizer = request.app.state.math_recognizer

    for block in blocks:
        crop_box = (int(block.bbox.x1), int(block.bbox.y1), int(block.bbox.x2), int(block.bbox.y2))
        crop = image.crop(crop_box)

        if block.type == "text" and text_ocr is not None:
            block.content, block.confidence = text_ocr.recognize(crop)
        elif block.type == "math" and math_recognizer is not None:
            block.content, block.confidence = math_recognizer.predict(crop)

    tex_source = assemble_latex(blocks)

    pdf_url = None
    try:
        pdf_bytes = compile_tex_to_pdf(tex_source, timeout_seconds=settings.pdf_compile_timeout_seconds)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        with open(os.path.join(RESULTS_DIR, pdf_filename), "wb") as f:
            f.write(pdf_bytes)
        pdf_url = f"/static/results/{pdf_filename}"
    except PDFCompilationError:
        pdf_url = None

    return ConvertResponse(tex_source=tex_source, pdf_url=pdf_url)
