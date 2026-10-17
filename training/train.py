"""
FH-NNUE Training Pipeline

Supports:
- Architecture ablations: B0, A1, A2, A3
- PSQT skip ablation: with/without PSQT
- Data-efficiency scaling experiments: 1%, 5%, 10%, 25%, 100%
- Checkpointing and automatic export to `.nnue`
"""

import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    generate_synthetic_dataset,
    load_dataset_from_csv,
    split_games_without_leakage,
    build_dataset_from_games,
)
from model import FHNNUE
from export import export_model_to_binary


def collate_fn_batched(batch):
    stm_flat = []
    stm_offsets = [0]
    nstm_flat = []
    nstm_offsets = [0]
    targets = []

    for stm, nstm, tgt in batch:
        stm_flat.extend(stm)
        stm_offsets.append(stm_offsets[-1] + len(stm))
        nstm_flat.extend(nstm)
        nstm_offsets.append(nstm_offsets[-1] + len(nstm))
        targets.append(tgt)

    stm_offsets.pop()  # Length is batch_size
    nstm_offsets.pop()

    return (
        torch.tensor(stm_flat, dtype=torch.long),
        torch.tensor(stm_offsets, dtype=torch.long),
        torch.tensor(nstm_flat, dtype=torch.long),
        torch.tensor(nstm_offsets, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for stm_idx, stm_off, nstm_idx, nstm_off, targets in loader:
        stm_idx = stm_idx.to(device, non_blocking=True)
        stm_off = stm_off.to(device, non_blocking=True)
        nstm_idx = nstm_idx.to(device, non_blocking=True)
        nstm_off = nstm_off.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model.forward_batched(stm_idx, stm_off, nstm_idx, nstm_off)
        loss = F.binary_cross_entropy_with_logits(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    with torch.no_grad():
        for stm_idx, stm_off, nstm_idx, nstm_off, targets in loader:
            stm_idx = stm_idx.to(device, non_blocking=True)
            stm_off = stm_off.to(device, non_blocking=True)
            nstm_idx = nstm_idx.to(device, non_blocking=True)
            nstm_off = nstm_off.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model.forward_batched(stm_idx, stm_off, nstm_idx, nstm_off)
            loss = F.binary_cross_entropy_with_logits(preds, targets)
            probs = torch.sigmoid(preds)
            mse = F.mse_loss(probs, targets)

            total_loss += loss.item() * len(targets)
            total_mse += mse.item() * len(targets)

    n = len(loader.dataset)
    return total_loss / n, total_mse / n


def run_experiment(
    arch_type: str = "A3",
    has_psqt: bool = True,
    data_fraction: float = 1.0,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 2e-3,
    patience: int = 2,
    csv_path: str = None,
    max_games: int = None,
    max_positions: int = None,
    output_model_path: str = "default.nnue",
    seed: int = 42,
):
    import json
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running FH-NNUE Experiment ===")
    print(f"Arch: {arch_type}, PSQT: {has_psqt}, Batch Size: {batch_size}, LR: {lr}, Device: {device}")

    # 1. Dataset loading & splitting
    if csv_path and os.path.exists(csv_path):
        print(f"Loading real positions from CSV: {csv_path} (max_games={max_games}, max_positions={max_positions})")
        games = load_dataset_from_csv(csv_path, max_games=max_games, max_positions=max_positions)
    else:
        print("Using synthetic dataset generator...")
        games = generate_synthetic_dataset(num_games=100, positions_per_game=25, seed=seed)

    train_games, val_games, test_games = split_games_without_leakage(games, 0.8, 0.1, 0.1, seed=seed)

    train_ds = build_dataset_from_games(train_games, data_fraction=data_fraction, seed=seed)
    val_ds = build_dataset_from_games(val_games, data_fraction=1.0, seed=seed)
    test_ds = build_dataset_from_games(test_games, data_fraction=1.0, seed=seed)

    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Use num_workers=2 to prevent CPU overload while keeping GPU fed
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_batched, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_batched, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_batched, num_workers=2, pin_memory=True)

    # 2. Model initialization
    model = FHNNUE(arch_type=arch_type, has_psqt=has_psqt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.time()
    history = []

    print("\nStarting Training with Overfitting Prevention and Per-Epoch Checkpoints:")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mse = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s]: "
            f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val MSE = {val_mse:.4f} | LR = {current_lr:.1e}"
        )

        # Save per-epoch checkpoint and .nnue model
        epoch_pt = f"checkpoint_epoch_{epoch:02d}.pt"
        epoch_nnue = f"model_epoch_{epoch:02d}.nnue"
        torch.save(model.state_dict(), epoch_pt)
        export_model_to_binary(model, epoch_nnue)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_checkpoint.pt")
            export_model_to_binary(model, output_model_path)
            print(f"  --> Saved new best model to {output_model_path} (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --> Val loss did not improve ({epochs_no_improve}/{patience} towards early stopping)")

        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "lr": current_lr,
            "elapsed_s": elapsed,
        })
        with open("training_stats.json", "w") as f:
            json.dump(history, f, indent=2)

        scheduler.step(val_loss)

        # Early stopping to prevent overfitting
        if epochs_no_improve >= patience:
            print(f"\n[Overfitting Prevention] Early stopping triggered at epoch {epoch}! Best epoch was {best_epoch}.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s. Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")

    # Load best model for test evaluation
    if os.path.exists("best_checkpoint.pt"):
        model.load_state_dict(torch.load("best_checkpoint.pt"))
    test_loss, test_mse = evaluate(model, test_loader, device)
    print(f"Final Test Evaluation on Unseen Test Games: Test Loss = {test_loss:.4f}, Test MSE = {test_mse:.4f}")

    return {
        "arch_type": arch_type,
        "has_psqt": has_psqt,
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mse": test_mse,
        "training_time_s": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="A3", choices=["B0", "A1", "A2", "A3"])
    parser.add_argument("--no-psqt", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--csv", type=str, default=None, help="Path to positions.csv")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--output", type=str, default="default.nnue")
    args = parser.parse_args()

    run_experiment(
        arch_type=args.arch,
        has_psqt=not args.no_psqt,
        data_fraction=args.fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        csv_path=args.csv,
        max_games=args.max_games,
        max_positions=args.max_positions,
        output_model_path=args.output,
    )
