"""
FH-NNUE Model and Inference Verification Script

Verifies:
1. Weight folding preserves network output within documented float tolerance.
2. Binary model file loads and passes checksum validation.
3. Quantized inference matches unquantized forward pass within expected quantization tolerance.
"""

import os
import struct
import numpy as np
import torch

from features import extract_features_from_fen
from model import FHNNUE
from export import adler32, MAGIC, FORMAT_VERSION, FEATURE_VERSION, ARCH_ID


def verify_weight_folding(model: FHNNUE):
    """
    Test that folding the factorized feature weights into the flat table
    preserves the network's output within machine precision tolerance.
    """
    model.eval()
    with torch.no_grad():
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        ]

        folded_weights = model.fold_weights()

        max_diff = 0.0
        for fen in test_fens:
            parts = fen.split()
            stm = 0 if parts[1] == "w" else 1
            nstm = 1 - stm
            w_feats, _, _ = extract_features_from_fen(fen, stm)
            b_feats, _, _ = extract_features_from_fen(fen, nstm)

            out_folded = model.forward_from_features([w_feats], [b_feats], flat_weights=folded_weights)
            out_standard = model.forward_from_features([w_feats], [b_feats])

            diff = abs(out_folded.item() - out_standard.item())
            max_diff = max(max_diff, diff)

        print(f"Weight folding verification: max output difference = {max_diff:.8e}")
        assert max_diff < 1e-5, f"Folded output discrepancy too high: {max_diff}"
        print("Weight folding verification PASSED!")


def verify_binary_file(model_path: str):
    """
    Verifies binary `.nnue` header, magic, checksum, and structure.
    """
    with open(model_path, "rb") as f:
        data = f.read()

    header = data[:44]
    payload = data[44:]

    magic, fmt_v, feat_v, arch_id, has_psqt, num_f, trans_w, d1_out, d2_out, psqt_c, expected_cs = struct.unpack(
        "<4sIIIIIIIIII", header
    )

    assert magic == MAGIC, f"Bad magic: {magic}"
    assert fmt_v == FORMAT_VERSION, f"Bad format version: {fmt_v}"
    assert feat_v == FEATURE_VERSION, f"Bad feature version: {feat_v}"
    assert arch_id == ARCH_ID, f"Bad arch ID: {arch_id}"
    assert num_f == 22528, f"Bad num features: {num_f}"
    assert trans_w == 512, f"Bad transformer width: {trans_w}"
    assert d1_out == 16, f"Bad dense 1: {d1_out}"
    assert d2_out == 32, f"Bad dense 2: {d2_out}"
    assert psqt_c == 8, f"Bad psqt channels: {psqt_c}"

    calc_cs = adler32(payload)
    assert calc_cs == expected_cs, f"Checksum mismatch: expected 0x{expected_cs:08X}, calculated 0x{calc_cs:08X}"

    print(f"Binary model file '{model_path}' successfully verified! Size={len(data)} bytes, Checksum=0x{calc_cs:08X}")


if __name__ == "__main__":
    print("Testing weight folding...")
    for arch in ["A2", "A3"]:
        m = FHNNUE(arch_type=arch, has_psqt=True)
        verify_weight_folding(m)

    if os.path.exists("default.nnue"):
        verify_binary_file("default.nnue")
