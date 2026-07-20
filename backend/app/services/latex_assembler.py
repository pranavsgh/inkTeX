"""Takes ordered text/math blocks and wraps them into a complete .tex document string. (Pranav)"""

import re

from app.models.schemas import Block

_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
# Substituting in a single pass (rather than chained str.replace calls) so the backslashes
# introduced by escaping one character are never themselves re-escaped by a later one.
_ESCAPE_RE = re.compile("|".join(re.escape(char) for char in _LATEX_SPECIAL_CHARS))


def _escape_text(text: str) -> str:
    """Escape LaTeX special characters in recognized prose so it can't break compilation."""
    return _ESCAPE_RE.sub(lambda match: _LATEX_SPECIAL_CHARS[match.group()], text)


def assemble_latex(blocks: list[Block]) -> str:
    """Assemble ordered blocks into a compilable LaTeX document, wrapping math blocks in $...$."""
    body_parts = []
    for block in sorted(blocks, key=lambda b: b.order):
        if not block.content:
            continue
        if block.type == "math":
            body_parts.append(f"${block.content}$")
        else:
            body_parts.append(_escape_text(block.content))

    body = "\n\n".join(body_parts)

    return (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\begin{document}\n"
        f"{body}\n"
        "\\end{document}\n"
    )
