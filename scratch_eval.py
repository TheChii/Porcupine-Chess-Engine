import torch
import torch.nn as nn
import numpy as np
import chess

INPUT_SIZE = 768
L1_SIZE = 256
L2_SIZE = 64
SCALE = 400.0

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.embedding = nn.EmbeddingBag(INPUT_SIZE, L1_SIZE, mode='sum')
        self.fc1 = nn.Linear(L1_SIZE, L2_SIZE)
        self.fc2 = nn.Linear(L2_SIZE, 1)

    def forward(self, flat_indices, offsets):
        x = self.embedding(flat_indices, offsets)
        x = torch.clamp(x, 0.0, 1.0) ** 2
        x = self.fc1(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.fc2(x)
        return x.squeeze()

def load_binary(model, path):
    with open(path, 'rb') as f:
        # embedding
        emb_shape = model.embedding.weight.shape
        emb_size = np.prod(emb_shape)
        emb_data = np.frombuffer(f.read(emb_size * 4), dtype=np.float32).reshape(emb_shape).copy()
        model.embedding.weight.data = torch.from_numpy(emb_data)

        # fc1
        fc1_w_shape = model.fc1.weight.shape
        fc1_w_size = np.prod(fc1_w_shape)
        fc1_w_data = np.frombuffer(f.read(fc1_w_size * 4), dtype=np.float32).reshape(fc1_w_shape).copy()
        model.fc1.weight.data = torch.from_numpy(fc1_w_data)

        fc1_b_shape = model.fc1.bias.shape
        fc1_b_size = np.prod(fc1_b_shape)
        fc1_b_data = np.frombuffer(f.read(fc1_b_size * 4), dtype=np.float32).reshape(fc1_b_shape).copy()
        model.fc1.bias.data = torch.from_numpy(fc1_b_data)

        # fc2
        fc2_w_shape = model.fc2.weight.shape
        fc2_w_size = np.prod(fc2_w_shape)
        fc2_w_data = np.frombuffer(f.read(fc2_w_size * 4), dtype=np.float32).reshape(fc2_w_shape).copy()
        model.fc2.weight.data = torch.from_numpy(fc2_w_data)

        fc2_b_shape = model.fc2.bias.shape
        fc2_b_size = np.prod(fc2_b_shape)
        fc2_b_data = np.frombuffer(f.read(fc2_b_size * 4), dtype=np.float32).reshape(fc2_b_shape).copy()
        model.fc2.bias.data = torch.from_numpy(fc2_b_data)

def fen_to_indices(fen):
    board = chess.Board(fen)
    indices = []
    stm = board.turn
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None: continue
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

def evaluate(fen):
    model = NNUE()
    load_binary(model, "interrupted_network.bin")
    model.eval()
    
    indices = fen_to_indices(fen)
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    
    with torch.no_grad():
        raw = model(idx_tensor, offsets).item()
        
    return raw * 173.7178

if __name__ == "__main__":
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    q_sac_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1" # white missing queen
    b_q_sac_fen = "rnb1kbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1" # black missing queen
    w_sac_b_move = "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNB1KBNR b KQkq - 0 1"
    
    print(f"Start: cp={evaluate(start_fen):.1f}")
    print(f"W missing Q (W to move): cp={evaluate(q_sac_fen):.1f}")
    print(f"B missing Q (B to move): cp={evaluate(b_q_sac_fen):.1f}")
    print(f"W missing Q (B to move): cp={evaluate(w_sac_b_move):.1f}")
