"""Pydantic schemas shared across routers: bounding boxes, blocks, and API responses."""

from typing import Literal

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates of the source image."""

    x1: float
    y1: float
    x2: float
    y2: float


class Block(BaseModel):
    """A single detected region of a page, classified as text or math."""

    type: Literal["text", "math"]
    bbox: BoundingBox
    content: str
    confidence: float
    order: int


class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze."""

    blocks: list[Block]
    image_width: int
    image_height: int


class MathResponse(BaseModel):
    """Response body for POST /math."""

    latex: str
    confidence: float


class ConvertResponse(BaseModel):
    """Response body for POST /convert."""

    tex_source: str
    pdf_url: str | None
