import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    find_pairs,
    split_pairs,
    LudwigDataset,
)
from src.model import Ludwig


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for x, target in loader:
        x = x.to(device)
        target = target.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = model(x)
            loss = criterion(prediction, target)

            if training:
                loss.backward()

                # Prevent really large gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_checkpoint(path, model, optimizer, epoch, best_loss, channels):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "channels": channels,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="ludwig training")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(42)

    device = get_device()

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Dataset preparation
    pairs = find_pairs(
        "dataset/filter",
        "dataset/target"
    )

    train_pairs, val_pairs = split_pairs(pairs)

    train_dataset = LudwigDataset(train_pairs)
    val_dataset = LudwigDataset(val_pairs)

    print(f"Training songs: {len(train_pairs)}")
    print(f"Validation songs: {len(val_pairs)}")
    print(f"Training chunks: {len(train_dataset)}")
    print(f"Validation chunks: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = Ludwig(channels=args.channels).to(device)

    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    start_epoch = 1
    best_loss = float("inf")

    # Resume training
    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False
        )

        if checkpoint["channels"] != args.channels:
            raise ValueError("Checkpoint architecture mismatch")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]

        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    checkpoint_dir = Path("checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train: {train_loss:.6f} | "
            f"Validation: {val_loss:.6f}"
        )

        # Track the best validation loss before saving
        improved = val_loss < best_loss

        if improved:
            best_loss = val_loss

        # Save latest checkpoint
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_loss,
            args.channels
        )

        # Save best checkpoint
        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_loss,
                args.channels
            )

            print(f"New best validation loss: {best_loss:.6f}")

    print("Training finished!")
    print(f"Best validation loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
