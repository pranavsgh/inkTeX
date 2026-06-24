"""Resizes raw im2latex-100k images, builds the LaTeX vocabulary, and creates train/val/test splits."""


def preprocess(raw_data_dir: str, output_dir: str, image_size: tuple[int, int] = (256, 256)) -> None:
    """Resize images, tokenize formulas, build the vocabulary, and write out train/val/test splits."""
    raise NotImplementedError()


if __name__ == "__main__":
    raise NotImplementedError()
