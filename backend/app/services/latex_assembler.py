"""Takes ordered text/math blocks and wraps them into a complete .tex document string. (Shared)"""

from app.models.schemas import Block


def assemble_latex(blocks: list[Block]) -> str:
    """Assemble ordered blocks into a compilable LaTeX document, wrapping math blocks in $...$."""
    raise NotImplementedError()
