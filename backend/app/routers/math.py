"""POST /math — accepts a single cropped math-block image, returns its recognized LaTeX. (Pranav)"""

import io

from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.models.schemas import MathResponse

router = APIRouter(prefix="/math", tags=["math"])


@router.post("", response_model=MathResponse)
async def recognize_math(request: Request, file: UploadFile) -> MathResponse:
    """Run the math recognition model on an uploaded image of a single math block."""
    recognizer = request.app.state.math_recognizer
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Math recognition model is not loaded")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    latex, confidence = recognizer.predict(image)
    return MathResponse(latex=latex, confidence=confidence)
