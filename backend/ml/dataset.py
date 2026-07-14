"""PyTorch Dataset and DataLoader construction for the im2latex-100k dataset. (Pranav)"""

import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.augment import random_blur, random_contrast, random_noise, random_rotation
from ml.tokenizer import LatexTokenizer

AUGMENT_PROB = 0.5

_to_tensor = transforms.ToTensor()


def _augment(image: Image.Image) -> Image.Image:
    if random.random() < AUGMENT_PROB:
        image = random_rotation(image)
    if random.random() < AUGMENT_PROB:
        image = random_noise(image)
    if random.random() < AUGMENT_PROB:
        image = random_blur(image)
    if random.random() < AUGMENT_PROB:
        image = random_contrast(image)
    return image


class Im2LatexDataset(Dataset):
    """Loads (image, LaTeX formula) pairs from a preprocessed im2latex-100k split."""

    def __init__(self, data_dir: str, split: str = "train") -> None:
        """Index image paths and their corresponding LaTeX label strings for the given split."""
        self.split = split
        self.images_dir = os.path.join(data_dir, split, "images")
        self.tokenizer = LatexTokenizer(vocab_path=os.path.join(data_dir, "vocab.json"))

        self.examples: list[tuple[str, str]] = []
        labels_path = os.path.join(data_dir, split, "labels.txt")
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                image_name, formula = line.split("\t", 1)
                self.examples.append((image_name, formula))

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (image_tensor, token_id_tensor) pair at the given index."""
        image_name, formula = self.examples[idx]
        with Image.open(os.path.join(self.images_dir, image_name)) as image:
            image = image.convert("L")
            if self.split == "train":
                image = _augment(image)
            image_tensor = _to_tensor(image)

        token_ids = torch.tensor(self.tokenizer.encode(formula), dtype=torch.long)
        return image_tensor, token_ids


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack fixed-size images and right-pad variable-length token sequences with <pad> (id 0)."""
    images, token_seqs = zip(*batch)
    images = torch.stack(images)

    max_len = max(seq.size(0) for seq in token_seqs)
    padded = torch.zeros(len(token_seqs), max_len, dtype=torch.long)
    for i, seq in enumerate(token_seqs):
        padded[i, : seq.size(0)] = seq

    return images, padded


def build_dataloader(data_dir: str, split: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Construct a DataLoader over Im2LatexDataset for the given split."""
    dataset = Im2LatexDataset(data_dir, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)
