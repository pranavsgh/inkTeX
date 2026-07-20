"""Image resizing shared between training-time preprocessing and inference.

MathRecognizer must letterbox inference-time crops to the exact same size
training used (see scripts/preprocess_data.py) — CNNEncoder's positional
embedding is sized for a fixed sequence length, and even setting that aside,
a model only produces sane output at the input scale it was trained on.
"""

from PIL import Image, ImageOps


def letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize preserving aspect ratio, then pad with white to exactly `size`."""
    image = image.convert("L")
    image = ImageOps.contain(image, size)
    canvas = Image.new("L", size, color=255)
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas
