import torch
from train import NNUE, ChessDataset, collate_fn, BATCH_SIZE, save_binary
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChessDataset("dataset.txt", max_samples=40960*5)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    model = NNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(1):
        total_loss = 0
        for flat_indices, offsets, targets in loader:
            flat_indices = flat_indices.to(device)
            offsets = offsets.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(flat_indices, offsets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss: {total_loss / len(loader):.4f}")
    save_binary(model, "test_network.bin")

if __name__ == "__main__":
    run_test()
