"""Application settings, loaded from environment variables / .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the inkTeX backend.

    Values are sourced from environment variables, falling back to the
    defaults below. See .env.example for the full list of overridable keys.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    allowed_origins: str = "http://localhost:5173"

    math_model_weights_path: str = "weights/math_model.pt"
    layout_model_weights_path: str = "weights/yolov8_layout.pt"
    text_ocr_model_name: str = "microsoft/trocr-base-handwritten"

    device: str = "cpu"

    pdflatex_bin: str = "pdflatex"
    pdf_compile_timeout_seconds: int = 30


settings = Settings()
