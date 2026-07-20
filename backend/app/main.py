"""FastAPI application entrypoint: app instance, CORS, lifespan, and router mounting."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import analyze, convert, math
from app.services.math_recognizer import MathRecognizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models into memory on startup and release them on shutdown."""
    # TODO(Mutha): load layout detection (YOLOv8) and text OCR (TrOCR) models here too,
    # once layout_detector.py / text_ocr.py are implemented.
    if os.path.exists(settings.math_model_weights_path) and os.path.exists(settings.math_vocab_path):
        app.state.math_recognizer = MathRecognizer(
            weights_path=settings.math_model_weights_path,
            vocab_path=settings.math_vocab_path,
            device=settings.device,
        )
    else:
        app.state.math_recognizer = None

    yield

    app.state.math_recognizer = None


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
