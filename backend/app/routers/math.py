"""POST /math — accepts a single cropped math-block image, returns its recognized LaTeX. (Pranav)"""

from fastapi import APIRouter, UploadFile

from app.models.schemas import MathResponse

router = APIRouter(prefix="/math", tags=["math"])


@router.post("", response_model=MathResponse)
async def recognize_math(file: UploadFile) -> MathResponse:
    """Run the math recognition model on an uploaded image of a single math block."""
    raise NotImplementedError()
