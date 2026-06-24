"""Image augmentations applied during training: rotation, noise, blur, contrast. (Pranav)"""

from PIL import Image


def random_rotation(image: Image.Image, max_degrees: float = 5.0) -> Image.Image:
    """Rotate the image by a random angle within +/- max_degrees."""
    raise NotImplementedError()


def random_noise(image: Image.Image, intensity: float = 0.05) -> Image.Image:
    """Add random Gaussian noise to the image at the given intensity."""
    raise NotImplementedError()


def random_blur(image: Image.Image, max_radius: float = 1.5) -> Image.Image:
    """Apply Gaussian blur with a randomly chosen radius up to max_radius."""
    raise NotImplementedError()


def random_contrast(image: Image.Image, factor_range: tuple[float, float] = (0.8, 1.2)) -> Image.Image:
    """Randomly adjust image contrast within the given factor range."""
    raise NotImplementedError()
