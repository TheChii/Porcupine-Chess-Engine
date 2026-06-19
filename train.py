import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import os
from tqdm import tqdm

# ============================================================
#  LOD-NNUE v2.2 (Nano) — Normalized Symmetric HalfKP
#  22,528 → 32 (16 Us + 16 Them) → 32 → 16 → 1
# ============================================================

NUM_KING_BUCKETS = 32
NUM_PIECE_TYPES  = 11   # 5 friendly (P,N,B,R,Q) + 6 enemy (P,N,B,R,Q,K)
NUM_SQUARES      = 64
INPUT_SIZE       = NUM_KING_BUCKETS * NUM_PIECE_TYPES * NUM_SQUARES  # 22,528
ACC_SIZE         = 16   # Per-perspective accumulator width
L1_SIZE          = 32   # ACC_SIZE * 2  (Us ++ Them)
L2_SIZE          = 32   # Layer 1 output
L3_SIZE          = 16   # Layer 2 output
SCALE            = 400.0
BATCH_SIZE       = 4096
LEARNING_RATE    = 0.002
PATIENCE         = 8
MAX_EPOCHS       = 200

dataset_name = "dataset.bin"

# ---- Dataset / DataLoader ----

record_dtype = np.dtype([
    ('target', np.float32),
    ('num_stm', np.uint8),
    ('num_nstm', np.uint8),
    ('stm_indices', np.int16, (32,)),
    ('nstm_indices', np.int16, (32,)),
])

class ChessDataset(Dataset):
    def __init__(self, file_path):
        print(f"Memory-mapping {file_path} for ultra-fast access...")
        self.data = np.memmap(file_path, dtype=record_dtype, mode='r')
        print(f"Mapped {len(self.data)} pre-compiled positions.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        
        stm_len = record['num_stm']
        nstm_len = record['num_nstm']
        
        # Extract the exact active indices for this position
        stm = record['stm_indices'][:stm_len]
        nstm = record['nstm_indices'][:nstm_len]
        
        # Casting to int64 creates a copy, safe for PyTorch
        return (
            torch.from_numpy(stm.astype(np.int64)),
            torch.from_numpy(nstm.astype(np.int64)),
            torch.tensor(record['target'], dtype=torch.float32),
        )

def collate_fn(batch):
    stm_list  = [b[0] for b in batch]
    nstm_list = [b[1] for b in batch]
    targets   = torch.stack([b[2] for b in batch])

    stm_offsets  = torch.tensor([0] + [len(s) for s in stm_list][:-1],
                                dtype=torch.long).cumsum(0)
    nstm_offsets = torch.tensor([0] + [len(n) for n in nstm_list][:-1],
                                dtype=torch.long).cumsum(0)

    return (torch.cat(stm_list), stm_offsets,
            torch.cat(nstm_list), nstm_offsets,
            targets)

# ---- Model ----

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        # Layer 0: shared accumulator  22,528 → 16  (per perspective)
        self.accumulator = nn.EmbeddingBag(INPUT_SIZE, ACC_SIZE, mode='sum')
        self.acc_bias    = nn.Parameter(torch.zeros(ACC_SIZE))

        # Layer 1: 32 → 32   (CReLU)
        self.fc1 = nn.Linear(ACC_SIZE * 2, L2_SIZE)
        # Layer 2: 32 → 16   (CReLU)
        self.fc2 = nn.Linear(L2_SIZE, L3_SIZE)
        # Layer 3: 16 → 1    (output)
        self.fc3 = nn.Linear(L3_SIZE, 1)

        # Initialization
        nn.init.uniform_(self.accumulator.weight, -0.1, 0.1)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, stm_idx, stm_off, nstm_idx, nstm_off):
        stm  = self.accumulator(stm_idx,  stm_off)  + self.acc_bias
        nstm = self.accumulator(nstm_idx, nstm_off)  + self.acc_bias

        # CReLU on accumulators
        stm  = torch.clamp(stm,  0.0, 1.0)
        nstm = torch.clamp(nstm, 0.0, 1.0)

        # Concatenate: [Us, Them]  → 32
        x = torch.cat([stm, nstm], dim=1)

        # Layer 1: CReLU
        x = self.fc1(x)
        x = torch.clamp(x, 0.0, 1.0)

        # Layer 2: CReLU
        x = self.fc2(x)
        x = torch.clamp(x, 0.0, 1.0)

        # Layer 3: output (raw logit for BCEWithLogitsLoss)
        x = self.fc3(x)
        return x.squeeze()

# ---- Training loop ----

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = ChessDataset(dataset_name)

    train_size = int(0.95 * len(dataset))
    val_size   = len(dataset) - train_size
    generator  = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator)

    pin     = device.type == "cuda"
    workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, pin_memory=pin,
        num_workers=workers, persistent_workers=(workers > 0))
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        collate_fn=collate_fn, pin_memory=pin,
        num_workers=workers, persistent_workers=(workers > 0))

    model     = NNUE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    os.makedirs("backups", exist_ok=True)

    # Resume from checkpoint if it exists
    start_epoch     = 0
    best_val_loss   = float('inf')
    patience_counter = 0

    if os.path.exists("checkpoint.pt"):
        print("Resuming from checkpoint...")
        ckpt = torch.load("checkpoint.pt", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_loss    = ckpt["best_val_loss"]
        patience_counter = ckpt["patience_counter"]
        print(f"Resumed from epoch {start_epoch}, best loss: {best_val_loss:.6f}")

    # Dry-run save/load sanity check
    print("Performing dry run check...")
    test_model = NNUE().to(device)
    torch.save(test_model.state_dict(), "test_save.pt")
    test_model.load_state_dict(torch.load("test_save.pt", weights_only=True))
    os.remove("test_save.pt")
    print("Dry run: Save/Load OK.")

    def save_checkpoint(epoch, interrupted=False):
        ckpt = {
            "epoch":            epoch,
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "scheduler":        scheduler.state_dict(),
            "scaler":           scaler.state_dict() if scaler else None,
            "best_val_loss":    best_val_loss,
            "patience_counter": patience_counter,
        }
        torch.save(ckpt, "checkpoint.pt")
        if epoch % 5 == 0:
            torch.save(ckpt, f"backups/checkpoint_epoch_{epoch}.pt")
        if interrupted:
            torch.save(model.state_dict(), "interrupted_model.pt")
            save_binary(model, "interrupted_network.bin")
            print("Saved interrupted model + binary.")

    epoch = start_epoch
    try:
        while epoch < start_epoch + MAX_EPOCHS:
            # --- train ---
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for i, (si, so, ni, no_, tgt) in enumerate(pbar):
                si  = si.to(device, non_blocking=True)
                so  = so.to(device, non_blocking=True)
                ni  = ni.to(device, non_blocking=True)
                no_ = no_.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(si, so, ni, no_)
                        loss = criterion(outputs, tgt)
                        
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(si, so, ni, no_)
                    loss = criterion(outputs, tgt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                train_loss += loss.item()
                pbar.set_postfix({'loss': train_loss / (i + 1)})

            # --- validate ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for si, so, ni, no_, tgt in val_loader:
                    si  = si.to(device, non_blocking=True)
                    so  = so.to(device, non_blocking=True)
                    ni  = ni.to(device, non_blocking=True)
                    no_ = no_.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)
                    
                    if scaler is not None:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model(si, so, ni, no_)
                            loss = criterion(outputs, tgt)
                    else:
                        outputs = model(si, so, ni, no_)
                        loss = criterion(outputs, tgt)
                        
                    val_loss += loss.item() * tgt.size(0)
            avg_val_loss = val_loss / len(val_dataset)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

            # --- best model ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
                save_binary(model, "best_network.bin")
                print("New best model saved!")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

            save_checkpoint(epoch)
            scheduler.step(avg_val_loss)
            epoch += 1

            if patience_counter >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs. Stopping.")
                save_binary(model, "final_network.bin")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
        save_checkpoint(epoch, interrupted=True)

# ---- Binary weight export ----
# Layout: accumulator weights, acc bias, fc1 w/b, fc2 w/b, fc3 w/b

def save_binary(model, path):
    expected_params = (INPUT_SIZE * ACC_SIZE  # accumulator weights
                       + ACC_SIZE             # accumulator bias
                       + L1_SIZE * L2_SIZE    # fc1 weights
                       + L2_SIZE              # fc1 bias
                       + L2_SIZE * L3_SIZE    # fc2 weights
                       + L3_SIZE              # fc2 bias
                       + L3_SIZE * 1          # fc3 weights
                       + 1)                   # fc3 bias
    expected_bytes = expected_params * 4  # float32

    with open(path, 'wb') as f:
        # Accumulator  [22528 × 16]
        acc_w = model.accumulator.weight.detach().cpu().numpy().astype(np.float32)
        f.write(acc_w.tobytes())

        # Accumulator bias [16]
        acc_b = model.acc_bias.detach().cpu().numpy().astype(np.float32)
        f.write(acc_b.tobytes())

        # FC1  [32 × 32]  +  [32]
        f.write(model.fc1.weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc1.bias.detach().cpu().numpy().astype(np.float32).tobytes())

        # FC2  [16 × 32]  +  [16]
        f.write(model.fc2.weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc2.bias.detach().cpu().numpy().astype(np.float32).tobytes())

        # FC3  [1 × 16]   +  [1]
        f.write(model.fc3.weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc3.bias.detach().cpu().numpy().astype(np.float32).tobytes())

    actual_bytes = os.path.getsize(path)
    assert actual_bytes == expected_bytes, (
        f"Binary size mismatch! Expected {expected_bytes:,} bytes, "
        f"got {actual_bytes:,} bytes")

if __name__ == "__main__":
    train()
