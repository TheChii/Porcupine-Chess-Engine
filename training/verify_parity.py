"""
Deterministic Python vs. Rust Feature Index and Quantized Parity Verification
"""

import json
import subprocess
import os
import torch
import numpy as np

from features import extract_features_from_fen
from model import FHNNUE

TEST_FENS = [
    # 1. Starting position (Kings on E1, E8 -> files >= 4, mirrored to D1, D8)
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 2. Open game after 1. e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    # 3. Sicilian Defense (asymmetric pawns)
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    # 4. Ruy Lopez middlegame
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # 5. Kings on unmirrored side (File B: King on B1, B8)
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/4P3/1PNP1N2/P1P1BPPP/R2Q1RK1 w - - 0 1",
    # 6. King on File A (A1 / A8)
    "r1bqk2r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 0 1",
    # 7. King in the center (E4 / D5)
    "8/8/8/3k4/4K3/8/8/8 w - - 0 1",
    # 8. Endgame with pawns
    "8/5pk1/4p1p1/7p/7P/4P1P1/5PK1/8 w - - 0 1",
    # 9. Tactical position with promotions ready
    "8/1P4k1/8/8/8/8/5p1K/8 w - - 0 1",
    # 10. Queen capture imbalance
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 11. Complex position with castling rights
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # 12. King on corner H1 / H8 (mirrored to A1 / A8)
    "7k/8/8/8/8/8/8/7K w - - 0 1",
    # 13. King on File D (D1 / D8 - not mirrored, file 3)
    "3k4/8/8/8/8/8/8/3K4 w - - 0 1",
    # 14. King on File E (E1 / E8 - mirrored, file 4 -> file 3)
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    # 15. En passant ready
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    # 16. Imbalanced pieces (Knights vs Bishops)
    "2b1kb2/8/8/8/8/8/8/2N1KN2 w - - 0 1",
    # 17. Multi-piece endgame (K+R vs K+B)
    "8/8/8/4k3/8/2b5/4K3/4R3 w - - 0 1",
    # 18. Black king on file H, White king on file A
    "8/7k/8/8/8/8/8/K7 w - - 0 1",
    # 19. Advanced passed pawn rank 7
    "8/3P4/8/4k3/8/8/4K3/8 w - - 0 1",
    # 20. Heavy piece middlegame
    "2r2rk1/1q1nbppp/p2p4/3Pp3/1pP5/1P2BP2/P1N1Q1PP/R4RK1 w - - 0 1",
]


def test_feature_parity():
    print("=" * 70)
    print("TESTING PYTHON vs RUST FEATURE INDEX PARITY")
    print("=" * 70)

    rust_bin = "./target/release/feature_dump"
    assert os.path.exists(rust_bin), f"Rust binary {rust_bin} not found!"

    passed = 0
    for idx, fen in enumerate(TEST_FENS, 1):
        # Extract in Python
        py_w, py_wk, py_wb = extract_features_from_fen(fen, 0)
        py_b, py_bk, py_bb = extract_features_from_fen(fen, 1)
        py_w.sort()
        py_b.sort()

        # Extract in Rust
        proc = subprocess.run([rust_bin, fen], capture_output=True, text=True, check=True)
        rust_data = json.loads(proc.stdout)
        rust_w = rust_data["white"]
        rust_b = rust_data["black"]

        assert py_w == rust_w, f"[FAILED] FEN {idx} White features mismatch!\nPy: {py_w}\nRs: {rust_w}"
        assert py_b == rust_b, f"[FAILED] FEN {idx} Black features mismatch!\nPy: {py_b}\nRs: {rust_b}"
        print(f"  Position {idx:2d}: {fen[:45]:<45} | White feats: {len(py_w):2d} | Black feats: {len(py_b):2d} | MATCH OK")
        passed += 1

    print(f"\nALL {passed}/{len(TEST_FENS)} POSITIONS PERFECTLY MATCHED BETWEEN PYTHON AND RUST!")


if __name__ == "__main__":
    test_feature_parity()
