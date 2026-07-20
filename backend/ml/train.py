"""Training loop for Im2LatexModel: loss, optimizer, scheduler, checkpointing, WandB logging. (Pranav)"""

import os

import torch
import torch.nn as nn

from ml.dataset import build_dataloader
from ml.model import Im2LatexModel
from ml.tokenizer import PAD_TOKEN, SPECIAL_TOKENS, LatexTokenizer

try:
    import wandb
except ImportError:
    wandb = None

PAD_ID = SPECIAL_TOKENS.index(PAD_TOKEN)


def _run_epoch(
    model: Im2LatexModel,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one pass over `loader`; trains if `optimizer` is given, otherwise just evaluates."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    with torch.enable_grad() if is_train else torch.no_grad():
        for images, tokens in loader:
            images, tokens = images.to(device), tokens.to(device)
            # Teacher forcing: predict token[t+1] from tokens[:t+1].
            input_tokens, target_labels = tokens[:, :-1], tokens[:, 1:]

            logits = model(images, input_tokens)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    embed_dim: int = 512,
    checkpoint_every: int = 5,
    device: str | None = None,
    use_wandb: bool = True,
) -> None:
    """Run the full training loop, periodically checkpointing and logging metrics to WandB."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = LatexTokenizer(vocab_path=os.path.join(data_dir, "vocab.json"))
    train_loader = build_dataloader(data_dir, "train", batch_size, shuffle=True)
    val_loader = build_dataloader(data_dir, "val", batch_size, shuffle=False)

    model = Im2LatexModel(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    wandb_enabled = use_wandb and wandb is not None
    if wandb_enabled:
        wandb.init(
            project="inktex-math-recognition",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embed_dim": embed_dim,
                "vocab_size": tokenizer.vocab_size,
            },
        )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}/{epochs} - train_loss {train_loss:.4f} - val_loss {val_loss:.4f} - lr {current_lr:.2e}")
        if wandb_enabled:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

        if epoch % checkpoint_every == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt"),
            )

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
    )
