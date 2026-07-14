"""Image augmentations applied during training: rotation, noise, blur, contrast. (Pranav)"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _fill_color(image: Image.Image) -> int | tuple[int, int, int]:
    return 255 if image.mode == "L" else (255, 255, 255)


def random_rotation(image: Image.Image, max_degrees: float = 5.0) -> Image.Image:
    """Rotate the image by a random angle within +/- max_degrees."""
    angle = random.uniform(-max_degrees, max_degrees)
    return image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=_fill_color(image))


def random_noise(image: Image.Image, intensity: float = 0.05) -> Image.Image:
    """Add random Gaussian noise to the image at the given intensity."""
    array = np.asarray(image).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=intensity * 255.0, size=array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode=image.mode)


def random_blur(image: Image.Image, max_radius: float = 1.5) -> Image.Image:
    """Apply Gaussian blur with a randomly chosen radius up to max_radius."""
    radius = random.uniform(0, max_radius)
    return image.filter(ImageFilter.GaussianBlur(radius))


def random_contrast(image: Image.Image, factor_range: tuple[float, float] = (0.8, 1.2)) -> Image.Image:
    """Randomly adjust image contrast within the given factor range."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Contrast(image).enhance(factor)
