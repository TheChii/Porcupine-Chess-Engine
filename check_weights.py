import torch
from train import NNUE
model = NNUE()
ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
print("fc2 weight mean:", model.fc2.weight.mean().item())
print("fc2 weight std:", model.fc2.weight.std().item())
print("fc2 weight max:", model.fc2.weight.max().item())
print("fc2 weight min:", model.fc2.weight.min().item())

print("fc1 weight std:", model.fc1.weight.std().item())
print("embedding weight std:", model.embedding.weight.std().item())
