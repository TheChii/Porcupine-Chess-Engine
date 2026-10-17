"""
FH-NNUE Model Exporter

Converts a PyTorch FHNNUE model into the versioned binary format `.nnue`
consumed natively by the Rust chess engine loader.
"""

import struct
import torch
import numpy as np

from features import NUM_FEATURES
from model import FHNNUE, TRANSFORMER_WIDTH, DENSE1_OUT, DENSE2_OUT, PSQT_CHANNELS

MAGIC = b"FNUE"
FORMAT_VERSION = 1
FEATURE_VERSION = 1
ARCH_ID = 1

# Quantization scales
FT_SCALE = 255.0       # Feature transformer weight scale -> i16
DENSE_SCALE = 64.0     # Dense layer weight scale -> i8
OUTPUT_WEIGHT_SCALE = 128.0
PSQT_WEIGHT_SCALE = 64.0


def adler32(data: bytes) -> int:
    a = 1
    b = 0
    MOD = 65521
    for byte in data:
        a = (a + byte) % MOD
        b = (b + a) % MOD
    return (b << 16) | a


def export_model_to_binary(model: FHNNUE, output_path: str):
    """
    Folds weights and serializes the network into binary format.
    """
    model.eval()
    with torch.no_grad():
        # 1. Fold factorized weights into flat [22528, 512] table
        flat_weights = model.fold_weights().cpu().numpy()
        flat_biases = model.feature_biases.cpu().numpy()

        # Quantize feature transformer
        q_ft_weights = np.clip(np.round(flat_weights * FT_SCALE), -32768, 32767).astype(np.int16)
        q_ft_biases = np.clip(np.round(flat_biases * FT_SCALE), -32768, 32767).astype(np.int16)

        payload_bytes = bytearray()

        # 1. FT weights: [22528 * 512] i16
        payload_bytes.extend(q_ft_weights.tobytes())

        # 2. FT biases: [512] i16
        payload_bytes.extend(q_ft_biases.tobytes())

        # 3. PSQT (if enabled)
        if model.has_psqt:
            pw = model.psqt_weights.cpu().numpy()
            pb = model.psqt_biases.cpu().numpy()
            ph = model.psqt_head.weight.squeeze(0).cpu().numpy()

            q_pw = np.clip(np.round(pw * PSQT_WEIGHT_SCALE), -2147483648, 2147483647).astype(np.int32)
            q_pb = np.clip(np.round(pb * PSQT_WEIGHT_SCALE), -2147483648, 2147483647).astype(np.int32)
            q_ph = np.clip(np.round(ph * PSQT_WEIGHT_SCALE), -2147483648, 2147483647).astype(np.int32)

            payload_bytes.extend(q_pw.tobytes())
            payload_bytes.extend(q_pb.tobytes())
            payload_bytes.extend(q_ph.tobytes())

        # 4. Dense 1: weights [128, 1024] i8, biases [128] i32
        d1_w = model.dense1.weight.cpu().numpy()
        d1_b = model.dense1.bias.cpu().numpy()
        q_d1_w = np.clip(np.round(d1_w * DENSE_SCALE), -128, 127).astype(np.int8)
        q_d1_b = np.clip(np.round(d1_b * DENSE_SCALE * 127.0), -2147483648, 2147483647).astype(np.int32)

        payload_bytes.extend(q_d1_w.tobytes())
        payload_bytes.extend(q_d1_b.tobytes())

        # 5. Output: weights [128] i8, bias [1] i32
        out_w = model.output.weight.squeeze(0).cpu().numpy()
        out_b = model.output.bias.item()
        q_out_w = np.clip(np.round(out_w * OUTPUT_WEIGHT_SCALE), -128, 127).astype(np.int8)
        q_out_b = int(np.clip(round(out_b * OUTPUT_WEIGHT_SCALE * 127.0), -2147483648, 2147483647))

        payload_bytes.extend(q_out_w.tobytes())
        payload_bytes.extend(struct.pack("<i", q_out_b))

        # Checksum
        checksum = adler32(payload_bytes)

        # Header (44 bytes)
        has_psqt_val = 1 if model.has_psqt else 0
        header = struct.pack(
            "<4sIIIIIIIIII",
            MAGIC,
            FORMAT_VERSION,
            FEATURE_VERSION,
            ARCH_ID,
            has_psqt_val,
            NUM_FEATURES,
            TRANSFORMER_WIDTH,
            DENSE1_OUT,
            DENSE2_OUT,
            PSQT_CHANNELS,
            checksum,
        )

        with open(output_path, "wb") as f:
            f.write(header)
            f.write(payload_bytes)

    print(f"Exported model to {output_path} (header: {len(header)} bytes, payload: {len(payload_bytes)} bytes, total: {len(header) + len(payload_bytes)} bytes)")


if __name__ == "__main__":
    import os
    model = FHNNUE(arch_type="A3", has_psqt=True)
    if os.path.exists("best_checkpoint.pt"):
        print("Loading weights from best_checkpoint.pt...")
        ckpt = torch.load("best_checkpoint.pt", map_location="cpu")
        model.load_state_dict(ckpt)
    export_model_to_binary(model, "default.nnue")
