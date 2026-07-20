"""FastAPI application entrypoint: app instance, CORS, lifespan, and router mounting."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import analyze, convert, math
from app.services.layout_detector import LayoutDetector
from app.services.math_recognizer import MathRecognizer
from app.services.text_ocr import TextOcr


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models into memory on startup and release them on shutdown."""
    if os.path.exists(settings.math_model_weights_path) and os.path.exists(settings.math_vocab_path):
        app.state.math_recognizer = MathRecognizer(
            weights_path=settings.math_model_weights_path,
            vocab_path=settings.math_vocab_path,
            device=settings.device,
        )
    else:
        app.state.math_recognizer = None

    if os.path.exists(settings.layout_model_weights_path):
        app.state.layout_detector = LayoutDetector(
            weights_path=settings.layout_model_weights_path,
            device=settings.device,
        )
    else:
        app.state.layout_detector = None

    # TrOCR is loaded by name from the HuggingFace Hub rather than a local weights file,
    # so it's always attempted (it's a network/first-download failure, not a "not trained
    # yet" state like the other two models).
    try:
        app.state.text_ocr = TextOcr(model_name=settings.text_ocr_model_name, device=settings.device)
    except OSError:
        app.state.text_ocr = None

    yield

    app.state.math_recognizer = None
    app.state.layout_detector = None
    app.state.text_ocr = None


app = FastAPI(title="inkTeX API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(math.router)
app.include_router(analyze.router)
app.include_router(convert.router)
