import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import os
import time
from tqdm import tqdm
import multiprocessing as mp

# Constants
INPUT_SIZE = 768  # 2 * 6 * 64
HIDDEN_SIZE = 128
SCALE = 400.0
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 0.001

def fen_to_indices(fen):
    try:
        board = chess.Board(fen)
        indices = []
        stm = board.turn
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            pt = piece.piece_type - 1 
            pc = piece.color
            if stm == chess.WHITE:
                my_color = (pc == chess.WHITE)
                index_sq = sq
            else:
                my_color = (pc == chess.BLACK)
                index_sq = sq ^ 56
            color_idx = 0 if my_color else 1
            index = (color_idx * 6 * 64) + (pt * 64) + index_sq
            indices.append(index)
        return indices
    except:
        return []

def parse_line(line):
    parts = line.strip().split('|')
    if len(parts) < 3:
        return None
    fen = parts[0]
    score = float(parts[1])
    indices = fen_to_indices(fen)
    if not indices:
        return None
    # Target is centipawns scaled by 1/SCALE
    target = score / SCALE
    return (indices, target)

class ChessDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        self.samples = []
        print(f"Loading data from {file_path} using {mp.cpu_count()} cores...")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
        
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(parse_line, lines), total=len(lines), desc="Parsing FENs"))
        
        for r in results:
            if r:
                indices, target = r
                self.samples.append((torch.tensor(indices, dtype=torch.long), torch.tensor(target, dtype=torch.float)))
        
        print(f"Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    indices_list = [b[0] for b in batch]
    targets = torch.stack([b[1] for b in batch])
    
    offsets = [0] + [len(idx) for idx in indices_list]
    offsets = torch.tensor(offsets[:-1], dtype=torch.long).cumsum(dim=0)
    flat_indices = torch.cat(indices_list)
    
    return flat_indices, offsets, targets

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.embedding = nn.EmbeddingBag(INPUT_SIZE, HIDDEN_SIZE, mode='sum')
        self.fc = nn.Linear(HIDDEN_SIZE, 1)
        
        # Initialize embedding weights to keep the sum around 0.5 (center of CReLU)
        # Since there are typically 32 pieces on the board, mean should be 0.5 / 32
        nn.init.normal_(self.embedding.weight, mean=0.5 / 32, std=0.01)
        # Initialize output weights to be small
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, flat_indices, offsets):
        x = self.embedding(flat_indices, offsets)
        x = torch.clamp(x, 0.0, 1.0) # CReLU
        x = self.fc(x)
        return x.squeeze()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # For safety, let's first check if saving and loading works with 100 samples
    print("Performing dry run check...")
    test_dataset = ChessDataset('dataset_corrected.txt', max_samples=100)
    if len(test_dataset) == 0:
        print("Error: Could not load any samples. Check dataset.txt format.")
        return
    
    test_model = NNUE().to(device)
    torch.save(test_model.state_dict(), "test_save.pt")
    if os.path.exists("test_save.pt"):
        test_model.load_state_dict(torch.load("test_save.pt"))
        print("Dry run: Save/Load OK.")
        os.remove("test_save.pt")
    else:
        print("Error: Save failed.")
        return

    # Load full dataset (or a large portion)
    dataset = ChessDataset('dataset_corrected.txt')
    
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, pin_memory=True)

    model = NNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    
    # Backup directory
    os.makedirs("backups", exist_ok=True)

    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for i, (flat_indices, offsets, targets) in enumerate(pbar):
                flat_indices = flat_indices.to(device, non_blocking=True)
                offsets = offsets.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(flat_indices, offsets)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': train_loss / (i + 1)})

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for flat_indices, offsets, targets in val_loader:
                    flat_indices = flat_indices.to(device, non_blocking=True)
                    offsets = offsets.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(flat_indices, offsets)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.6f}")

            # Backup
            backup_path = f"backups/model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), backup_path)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pt")
                print("New best model saved!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        torch.save(model.state_dict(), "interrupted_model.pt")

    # Final save
    torch.save(model.state_dict(), "final_model.pt")
    save_binary(model, "network.bin")
    print("Training complete. Weights exported to network.bin")

def save_binary(model, path):
    model.cpu()
    with open(path, 'wb') as f:
        # 1. Input weights (EmbeddingBag)
        # Shape: [768, 128]
        weights = model.embedding.weight.data.numpy().astype(np.float32)
        f.write(weights.tobytes())
        
        # 2. Output weights (Linear)
        # Shape: [1, 128]
        out_weights = model.fc.weight.data.numpy().astype(np.float32)
        f.write(out_weights.tobytes())
        
        # 3. Output bias (Linear)
        # Shape: [1]
        out_bias = model.fc.bias.data.numpy().astype(np.float32)
        f.write(out_bias.tobytes())

if __name__ == "__main__":
    train()
