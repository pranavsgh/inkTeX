"""POST /analyze — accepts a full handwritten page image, returns ordered text/math blocks. (Mutha)"""

from fastapi import APIRouter, UploadFile

from app.models.schemas import AnalyzeResponse

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("", response_model=AnalyzeResponse)
async def analyze_page(file: UploadFile) -> AnalyzeResponse:
    """Run layout detection on the uploaded page and return ordered blocks with bboxes and types."""
    raise NotImplementedError()
