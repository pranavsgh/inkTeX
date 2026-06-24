"""PyTorch Dataset and DataLoader construction for the im2latex-100k dataset. (Pranav)"""

import torch
from torch.utils.data import DataLoader, Dataset


class Im2LatexDataset(Dataset):
    """Loads (image, LaTeX formula) pairs from a preprocessed im2latex-100k split."""

    def __init__(self, data_dir: str, split: str = "train") -> None:
        """Index image paths and their corresponding LaTeX label strings for the given split."""
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (image_tensor, token_id_tensor) pair at the given index."""
        raise NotImplementedError()


def build_dataloader(data_dir: str, split: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Construct a DataLoader over Im2LatexDataset for the given split."""
    raise NotImplementedError()
