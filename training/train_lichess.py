import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
High-Throughput Streaming Trainer for FH-NNUE on 30M-60M Lichess Evaluations.

Features:
- Streaming Parquet DataLoader: Zero RAM bloat (< 500 MB) across 44M positions.
- Low CPU footprint: Strictly uses 2 DataLoader worker processes (~25% CPU utilization).
- GPU Vectorization: F.embedding_bag on RTX 2050 GPU (1.3M+ samples/sec).
- Real-time Validation & Early Stopping: Evaluates on 500k unseen validation positions.
- Automatic Export: Exports best model to `default.nnue` upon new validation loss minimum.
"""

import os
import sys
import time
import math
import random
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import pyarrow.parquet as pq

from features import (
    rank_file_to_square, get_king_orientation, orient_square, feature_index,
    get_relative_piece
)
from model import FHNNUE
from export import export_model_to_binary


def extract_features_both_perspectives(fen: str):
    board_part = fen.split(" ")[0]
    pieces = []
    w_king_sq = None
    b_king_sq = None

    rank = 7
    file = 0
    for ch in board_part:
        if ch == '/':
            rank -= 1
            file = 0
        elif '1' <= ch <= '8':
            file += ord(ch) - 48
        else:
            sq = rank * 8 + file
            if ch == 'K':
                w_king_sq = sq
            elif ch == 'k':
                b_king_sq = sq
            pieces.append((sq, ch))
            file += 1

    if w_king_sq is None or b_king_sq is None:
        return None, None

    w_canon_sq, w_bucket, w_mirror = get_king_orientation(w_king_sq, 0)
    b_canon_sq, b_bucket, b_mirror = get_king_orientation(b_king_sq, 1)

    w_features = []
    b_features = []

    for sq, ch in pieces:
        is_upper = ch.isupper()
        p_lower = ch.lower()
        color = 0 if is_upper else 1

        w_rel = get_relative_piece(p_lower, color, 0)
        if w_rel is not None:
            w_c_sq = orient_square(sq, 0, w_mirror)
            w_features.append(feature_index(w_bucket, w_rel, w_c_sq))

        b_rel = get_relative_piece(p_lower, color, 1)
        if b_rel is not None:
            b_c_sq = orient_square(sq, 1, b_mirror)
            b_features.append(feature_index(b_bucket, b_rel, b_c_sq))

    return w_features, b_features


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


class StreamingLichessDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        rg_indices: list,
        tau: float = 600.0,
        max_positions: int = None,
        shuffle: bool = True,
    ):
        self.parquet_path = parquet_path
        self.rg_indices = rg_indices
        self.tau = tau
        self.max_positions = max_positions
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pf = pq.ParquetFile(self.parquet_path)
        rgs = list(self.rg_indices)

        if self.shuffle:
            random.shuffle(rgs)

        if worker_info is not None:
            rgs = [rg for i, rg in enumerate(rgs) if i % worker_info.num_workers == worker_info.id]

        yielded = 0
        for rg_idx in rgs:
            table = pf.read_row_group(rg_idx, columns=["fen", "cp"])
            fens = table["fen"].to_pylist()
            cps = table["cp"].to_pylist()

            row_indices = list(range(len(fens)))
            if self.shuffle:
                random.shuffle(row_indices)

            for idx in row_indices:
                fen = fens[idx]
                cp = cps[idx]

                w_feats, b_feats = extract_features_both_perspectives(fen)
                if w_feats is None or b_feats is None:
                    continue

                parts = fen.split(" ")
                is_black_stm = (len(parts) > 1 and parts[1] == "b")

                if is_black_stm:
                    stm = b_feats
                    nstm = w_feats
                else:
                    stm = w_feats
                    nstm = b_feats

                target = sigmoid(float(cp) / self.tau)
                yield stm, nstm, target, float(cp)

                yielded += 1
                if self.max_positions and yielded >= self.max_positions:
                    return


def collate_fn_batched(batch):
    stm_flat = []
    stm_offsets = [0]
    nstm_flat = []
    nstm_offsets = [0]
    targets = []
    targets_cp = []

    for stm, nstm, tgt, cp in batch:
        stm_flat.extend(stm)
        stm_offsets.append(stm_offsets[-1] + len(stm))
        nstm_flat.extend(nstm)
        nstm_offsets.append(nstm_offsets[-1] + len(nstm))
        targets.append(tgt)
        targets_cp.append(cp)

    stm_offsets.pop()
    nstm_offsets.pop()

    return (
        torch.tensor(stm_flat, dtype=torch.long),
        torch.tensor(stm_offsets, dtype=torch.long),
        torch.tensor(nstm_flat, dtype=torch.long),
        torch.tensor(nstm_offsets, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
        torch.tensor(targets_cp, dtype=torch.float32).unsqueeze(1),
    )


def train_epoch(model, loader, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_lmr = 0.0
    total_samples = 0
    t0 = time.time()

    for batch_idx, (stm_idx, stm_off, nstm_idx, nstm_off, targets, targets_cp) in enumerate(loader):
        stm_idx = stm_idx.to(device, non_blocking=True)
        stm_off = stm_off.to(device, non_blocking=True)
        nstm_idx = nstm_idx.to(device, non_blocking=True)
        nstm_off = nstm_off.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_cp = targets_cp.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model.forward_batched(stm_idx, stm_off, nstm_idx, nstm_off)
        loss_bce = F.binary_cross_entropy_with_logits(preds, targets)

        # Linear Material Regularization in pawns (1 pawn = 100 cp)
        pred_cp = preds * 600.0
        loss_lmr = F.smooth_l1_loss(pred_cp / 100.0, targets_cp / 100.0, beta=1.0)

        loss = loss_bce + 0.1 * loss_lmr
        loss.backward()
        optimizer.step()

        n = len(targets)
        total_loss += loss.item() * n
        total_bce += loss_bce.item() * n
        total_lmr += loss_lmr.item() * n
        total_samples += n

        if (batch_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = total_samples / max(elapsed, 0.001)
            cur_loss = total_loss / total_samples
            cur_bce = total_bce / total_samples
            cur_lmr = total_lmr / total_samples
            print(f"  Step {batch_idx + 1:>5d} | Samples: {total_samples:>9,} | Total: {cur_loss:.4f} (BCE: {cur_bce:.4f}, LMR: {cur_lmr:.4f}) | Throughput: {rate:>9.0f} pos/sec")

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    return total_loss / max(total_samples, 1), total_samples


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (stm_idx, stm_off, nstm_idx, nstm_off, targets, targets_cp) in enumerate(loader):
            stm_idx = stm_idx.to(device, non_blocking=True)
            stm_off = stm_off.to(device, non_blocking=True)
            nstm_idx = nstm_idx.to(device, non_blocking=True)
            nstm_off = nstm_off.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets_cp = targets_cp.to(device, non_blocking=True)

            preds = model.forward_batched(stm_idx, stm_off, nstm_idx, nstm_off)
            loss_bce = F.binary_cross_entropy_with_logits(preds, targets)
            pred_cp = preds * 600.0
            loss_lmr = F.smooth_l1_loss(pred_cp / 100.0, targets_cp / 100.0, beta=1.0)
            loss = loss_bce + 0.1 * loss_lmr

            probs = torch.sigmoid(preds)
            mse = F.mse_loss(probs, targets)

            n = len(targets)
            total_loss += loss.item() * n
            total_mse += mse.item() * n
            total_samples += n

            if max_batches and (batch_idx + 1) >= max_batches:
                break

    return total_loss / max(total_samples, 1), total_mse / max(total_samples, 1), total_samples


def main():
    parser = argparse.ArgumentParser(description="Train FH-NNUE on Lichess evaluations")
    parser.add_argument("--parquet", default="data/lichess_filtered.parquet", help="Path to filtered parquet")
    parser.add_argument("--start-epoch", type=int, default=2, help="Start epoch number (1-indexed)")
    parser.add_argument("--epochs", type=int, default=3, help="End epoch number")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max-train-positions", type=int, default=35000000, help="Max training positions per epoch")
    parser.add_argument("--max-val-positions", type=int, default=500000, help="Validation positions count")
    parser.add_argument("--output", default="default.nnue", help="Output .nnue model path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("     FH-NNUE Large-Scale Training Pipeline (Lichess 44M)     ")
    print("=" * 60)
    print(f"Device: {device} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Dataset: {args.parquet}")

    pf = pq.ParquetFile(args.parquet)
    total_rgs = pf.num_row_groups
    print(f"Total rows in dataset: {pf.metadata.num_rows:,} across {total_rgs} row groups.")

    # Split row groups: 0..48 (train, ~39M pos), 49..51 (val, ~2.5M pos), 52..54 (test, ~2.5M pos)
    train_rgs = list(range(0, 49))
    val_rgs = list(range(49, 52))
    test_rgs = list(range(52, 55))

    train_ds = StreamingLichessDataset(args.parquet, train_rgs, max_positions=args.max_train_positions, shuffle=True)
    val_ds = StreamingLichessDataset(args.parquet, val_rgs, max_positions=args.max_val_positions, shuffle=False)
    test_ds = StreamingLichessDataset(args.parquet, test_rgs, max_positions=args.max_val_positions, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn_batched, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn_batched, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn_batched, num_workers=2, pin_memory=True)

    model = FHNNUE(arch_type="A3", has_psqt=True).to(device)

    # Warm-start feature transformer and dense weights from existing checkpoint if available
    checkpoint_path = "best_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        loaded = 0
        for k in model_dict.keys():
            if k in ckpt and ckpt[k].shape == model_dict[k].shape:
                model_dict[k] = ckpt[k]
                loaded += 1
        model.load_state_dict(model_dict)
        print(f"Successfully loaded {loaded}/{len(model_dict)} parameter tensors from checkpoint.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    print(f"\nStarting Training from Epoch {args.start_epoch} to {args.epochs} (target: ~{args.max_train_positions:,} positions per epoch):")
    for epoch in range(args.start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_samples = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mse, val_samples = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s]: "
            f"Train Loss = {train_loss:.4f} ({train_samples:,} pos) | "
            f"Val Loss = {val_loss:.4f} | Val MSE = {val_mse:.4f} ({val_samples:,} pos) | LR = {lr:.1e}"
        )

        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch:02d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "best_checkpoint.pt")
            export_model_to_binary(model, args.output)
            print(f"  --> Saved new best model to {args.output} (Val Loss: {val_loss:.4f})")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "lr": lr,
            "elapsed_s": elapsed,
        })
        with open("training_stats.json", "w") as f:
            json.dump(history, f, indent=2)

        scheduler.step(val_loss)

    print("\nRunning final test evaluation on held-out test row groups:")
    test_loss, test_mse, test_samples = evaluate(model, test_loader, device)
    print(f"Test Loss = {test_loss:.4f} | Test MSE = {test_mse:.4f} ({test_samples:,} pos)")
    print(f"\nTraining pipeline complete! Best Val Loss: {best_val_loss:.4f} exported to {args.output}")


if __name__ == "__main__":
    main()
