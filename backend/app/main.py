"""FastAPI application entrypoint: app instance, CORS, lifespan, and router mounting."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import analyze, convert, math


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models into memory on startup and release them on shutdown."""
    raise NotImplementedError()
    yield


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
