import torch
import torch.nn as nn
import chess
import numpy as np
import os

# Same architecture as in train.py
INPUT_SIZE = 768
HIDDEN_SIZE = 128
SCALE = 400.0

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.embedding = nn.EmbeddingBag(INPUT_SIZE, HIDDEN_SIZE, mode='sum')
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, indices_list):
        offsets = [0] + [len(idx) for idx in indices_list]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        flat_indices = torch.cat(indices_list)
        x = self.embedding(flat_indices, offsets)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.fc(x)
        return x.squeeze()

def fen_to_indices(fen):
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

def test_model(model_path, fens):
    model = NNUE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print(f"{'FEN':<60} | {'Prediction (cp)':<15}")
    print("-" * 80)
    
    for fen in fens:
        indices = fen_to_indices(fen)
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        with torch.no_grad():
            output = model([indices_tensor])
            # output is predicted score / SCALE
            score = output.item() * SCALE
            
            print(f"{fen:<60} | {score:>10.2f} cp")

if __name__ == "__main__":
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Startpos
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", # 1. e4
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # Ruy Lopez
        "8/8/8/4k3/8/8/4K3/8 w - - 0 1", # Drawish
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", # White slightly better (pawn up)
    ]
    
    model_to_test = "best_model.pt"
    if not os.path.exists(model_to_test):
        model_to_test = "final_model.pt"

    if os.path.exists(model_to_test):
        print(f"Testing model: {model_to_test}")
        test_model(model_to_test, test_fens)
    else:
        print("Model file not found. Train the model first.")
