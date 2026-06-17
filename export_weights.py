import torch
import torch.nn as nn
import numpy as np

INPUT_SIZE = 768
HIDDEN_SIZE = 128

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

def export_to_bin(model_path, out_path):
    model = NNUE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    with open(out_path, 'wb') as f:
        # 1. Input weights (EmbeddingBag)
        weights = model.embedding.weight.data.numpy().astype(np.float32)
        f.write(weights.tobytes())
        
        # 2. Output weights (Linear)
        out_weights = model.fc.weight.data.numpy().astype(np.float32)
        f.write(out_weights.tobytes())
        
        # 3. Output bias (Linear)
        out_bias = model.fc.bias.data.numpy().astype(np.float32)
        f.write(out_bias.tobytes())
    print(f"Exported {model_path} to {out_path}")

if __name__ == "__main__":
    import os
    if os.path.exists("best_model.pt"):
        export_to_bin("best_model.pt", "network.bin")
    elif os.path.exists("final_model.pt"):
        export_to_bin("final_model.pt", "network.bin")
    else:
        print("No model found to export.")
