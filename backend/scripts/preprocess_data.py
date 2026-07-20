"""Resizes raw im2latex-100k images, builds the LaTeX vocabulary, and creates train/val/test splits."""

import os

from PIL import Image, ImageOps

from ml.tokenizer import LatexTokenizer

# Expected raw layout (standard im2latex-100k distribution):
#   raw_data_dir/formula_images/<image_name>
#   raw_data_dir/im2latex_formulas.norm.lst      (one formula per line, indexed by line number)
#   raw_data_dir/im2latex_{train,validate,test}.lst  (lines: "<image_name> <formula_idx>")
RAW_SPLIT_FILES = {
    "train": "im2latex_train.lst",
    "val": "im2latex_validate.lst",
    "test": "im2latex_test.lst",
}
FORMULAS_FILE = "im2latex_formulas.norm.lst"
RAW_IMAGES_DIR = "formula_images"


def _load_formulas(raw_data_dir: str) -> list[str]:
    path = os.path.join(raw_data_dir, FORMULAS_FILE)
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _load_split_pairs(raw_data_dir: str, split: str) -> list[tuple[str, int]]:
    path = os.path.join(raw_data_dir, RAW_SPLIT_FILES[split])
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name, formula_idx = line.split()
            pairs.append((image_name, int(formula_idx)))
    return pairs


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize preserving aspect ratio, then pad with white to exactly `size`."""
    image = image.convert("L")
    image = ImageOps.contain(image, size)
    canvas = Image.new("L", size, color=255)
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def preprocess(raw_data_dir: str, output_dir: str, image_size: tuple[int, int] = (256, 256)) -> None:
    """Resize images, tokenize formulas, build the vocabulary, and write out train/val/test splits."""
    formulas = _load_formulas(raw_data_dir)
    raw_images_dir = os.path.join(raw_data_dir, RAW_IMAGES_DIR)

    train_pairs = _load_split_pairs(raw_data_dir, "train")

    tokenizer = LatexTokenizer()
    tokenizer.build_vocab([formulas[idx] for _, idx in train_pairs])
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "vocab.json"))

    for split in RAW_SPLIT_FILES:
        pairs = train_pairs if split == "train" else _load_split_pairs(raw_data_dir, split)
        split_images_dir = os.path.join(output_dir, split, "images")
        os.makedirs(split_images_dir, exist_ok=True)

        labels_path = os.path.join(output_dir, split, "labels.txt")
        with open(labels_path, "w", encoding="utf-8") as labels_file:
            for image_name, formula_idx in pairs:
                with Image.open(os.path.join(raw_images_dir, image_name)) as image:
                    resized = _letterbox(image, image_size)
                    resized.save(os.path.join(split_images_dir, image_name))
                labels_file.write(f"{image_name}\t{formulas[formula_idx]}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_data_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    preprocess(args.raw_data_dir, args.output_dir, (args.width, args.height))
