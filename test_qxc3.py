import torch
from scratch_eval import evaluate
fen = "rnb1kbnr/ppp1pppp/8/8/4P3/2q5/PPPP1PPP/R1B1KBNR w KQkq - 0 5"
print(f"cp={evaluate(fen):.1f}")
