"""Training loop for Im2LatexModel: loss, optimizer, scheduler, checkpointing, WandB logging. (Pranav)"""


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
) -> None:
    """Run the full training loop, periodically checkpointing and logging metrics to WandB."""
    raise NotImplementedError()


if __name__ == "__main__":
    raise NotImplementedError()
