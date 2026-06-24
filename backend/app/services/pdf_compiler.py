"""Takes a .tex source string, writes it to a temp file, runs pdflatex, and returns PDF bytes. (Pranav)"""


def compile_tex_to_pdf(tex_source: str, timeout_seconds: int = 30) -> bytes:
    """Write tex_source to a temporary .tex file, invoke pdflatex, and return the resulting PDF bytes."""
    raise NotImplementedError()
