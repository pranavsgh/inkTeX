"""POST /convert — master endpoint: analyze page, route each block, assemble .tex, compile PDF. (Shared)"""

from fastapi import APIRouter, UploadFile

from app.models.schemas import ConvertResponse

router = APIRouter(prefix="/convert", tags=["convert"])


@router.post("", response_model=ConvertResponse)
async def convert_page(file: UploadFile) -> ConvertResponse:
    """Run the full pipeline on an uploaded page: analyze -> recognize blocks -> assemble -> compile."""
    raise NotImplementedError()
