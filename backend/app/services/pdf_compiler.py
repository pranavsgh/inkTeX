"""Takes a .tex source string, writes it to a temp file, runs pdflatex, and returns PDF bytes. (Pranav)"""

import os
import subprocess
import tempfile

from app.config import settings


class PDFCompilationError(Exception):
    """Raised when pdflatex fails or times out. `log` holds pdflatex's stdout/stderr for debugging."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


def compile_tex_to_pdf(tex_source: str, timeout_seconds: int = 30) -> bytes:
    """Write tex_source to a temporary .tex file, invoke pdflatex, and return the resulting PDF bytes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tex_path = os.path.join(tmp_dir, "doc.tex")
        pdf_path = os.path.join(tmp_dir, "doc.pdf")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_source)

        try:
            result = subprocess.run(
                [
                    settings.pdflatex_bin,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    tmp_dir,
                    tex_path,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            log = (exc.stdout or "") + (exc.stderr or "")
            raise PDFCompilationError(f"pdflatex timed out after {timeout_seconds}s", log=log) from exc
        except FileNotFoundError as exc:
            raise PDFCompilationError(f"pdflatex binary not found: {settings.pdflatex_bin}") from exc

        if result.returncode != 0 or not os.path.exists(pdf_path):
            raise PDFCompilationError(
                f"pdflatex exited with code {result.returncode}",
                log=result.stdout + result.stderr,
            )

        with open(pdf_path, "rb") as f:
            return f.read()
