# FH-NNUE: Factorized, Horizontally Canonicalized Neural Chess Evaluation in Rust

**FH-NNUE** is an experimental neural evaluation system implemented natively in Rust for the **Porcupine** chess engine, featuring a factorized feature transformer, horizontal canonicalization, an optional PSQT skip connection, and a Python (PyTorch) training and export pipeline.

---

## 1. Architecture Overview

### 1.1 Position-to-Feature Mapping (HalfKAv2 with Horizontal Symmetry)
- **Perspectives**: Evaluated independently from White and Black perspectives and combined into a side-to-move score.
- **Canonical King Buckets**: The perspective's own king square defines its king bucket. The board is horizontally reflected (`sq ^ 7`) whenever the king is on files E–H (files 4–7), ensuring the king always resides on files A–D (files 0–3). For Black, the board is vertically flipped (`sq ^ 56`) first. This yields **32 canonical king buckets**.
- **Piece Categories (11 Relative Categories)**:
  - Friendly: Pawn (0), Knight (1), Bishop (2), Rook (3), Queen (4).
  - Enemy: Pawn (5), Knight (6), Bishop (7), Rook (8), Queen (9), King (10).
  - Friendly king is omitted as an ordinary piece feature because it defines the king bucket.
- **Feature Dimension**: `32 king buckets × 11 categories × 64 squares = 22,528` ordinary features per perspective.
- **Castling Symmetry**: Castling rights are explicitly swapped (kingside ↔ queenside) upon horizontal reflection to eliminate false equivalence while preserving board symmetry.

### 1.2 Factorized Feature Transformer
During training, statistical strength is shared across king buckets via additive factorization:
\[
W(k, p, s) = W_{\text{global}}(p, s) + W_{\text{rel}}(p, \Delta r, \Delta f) + W_{\text{zone}}(Z(k), p, s) + W_{\text{local}}(k, p, s)
\]
Where:
- \(k\): Canonical king bucket (0..31).
- \(p\): Relative piece category (0..10).
- \(s\): Canonical square (0..63).
- \(\Delta r, \Delta f\): Piece displacement relative to the king (\(15 \times 15 = 225\) relative offsets).
- \(Z(k)\): Coarse king zone (4 zones: Castled corner, Queenside/center, Middle ranks, Advanced).
- \(W_{\text{global}}\): Global piece-square weights shared across all king buckets.
- \(W_{\text{rel}}\): Weights shared by relative displacement to the king.
- \(W_{\text{zone}}\): Coarse king-zone corrections.
- \(W_{\text{local}}\): Bucket-specific residual corrections.

**Zero-Overhead Production Inference**:
During export, factorized components are folded into a single flat `[22528, 512]` table:
\[
W_{\text{flat}}(k, p, s) = W_{\text{global}}(p, s) + W_{\text{rel}}(p, \Delta r, \Delta f) + W_{\text{zone}}(Z(k), p, s) + W_{\text{local}}(k, p, s)
\]
The engine performs direct flat table lookups with zero runtime overhead or factorization latency during search.

### 1.3 Optional PSQT Skip Connection
- 8 signed 32-bit channels per perspective.
- Accumulates piece-square values directly during incremental updates.
- Linearly projected and combined into the final evaluation score.

### 1.4 Quantized Neural Network Head
- **Input**: 1,024 dimensions (`[stm_accumulator, nstm_accumulator]`).
- **Activation**: Squared Clipped ReLU (SCReLU):
  \[
  \text{SCReLU}(u) = \left\lfloor \frac{\text{clamp}(u, 0, 255)^2}{512} \right\rfloor \in [0, 126]
  \]
- **Dense Layer 1**: 1,024 → 16 (`i8` weights, `i32` biases).
- **Dense Layer 2**: 16 → 32 (`i8` weights, `i32` biases).
- **Output Layer**: 32 → 1 (`i8` weights, `i32` bias).
- **Safe Range**: Intermediate multiplications use 32-bit integer arithmetic with explicit saturation and clamping.

---

## 2. File Organization

```
├── src/
│   ├── nnue/
│   │   ├── accumulator.rs    # Fixed-size 512-dim accumulators and incremental updates
│   │   ├── eval.rs           # Evaluation interface and global model manager
│   │   ├── features.rs       # Canonical king buckets, orientation, 11 piece categories
│   │   ├── mod.rs            # Module root and re-exports
│   │   ├── model.rs          # Versioned binary model loader (.nnue) & validator
│   │   ├── network.rs        # SCReLU, quantized dense head (1024->16->32->1)
│   │   └── tests.rs          # 38 unit & regression tests
│   ├── eval/                 # Tapered HCE with seamless NNUE integration & fallback
│   ├── search/               # Alpha-beta negamax with stack-allocated accumulators
│   ├── uci/                  # UCI protocol with UseNNUE and EvalFile options
│   └── bin/
│       └── nnue_bench.rs     # Performance benchmark suite
├── training/
│   ├── dataset.py            # Dataset loader, game-level splitting, blended target
│   ├── export.py             # Binary model exporter (.nnue) with Adler-32 checksum
│   ├── features.py           # Bit-exact Python feature extraction
│   ├── model.py              # PyTorch model with vectorized weight folding
│   ├── run_experiments.py    # Experimental ablation suite
│   ├── train.py              # Training script supporting data fraction scaling
│   └── verify.py             # Bit-level verification of weight folding and binary file
└── default.nnue              # Pretrained FH-NNUE model binary
```

---

## 3. Build & Test Commands

### 3.1 Build the Engine
```bash
cargo build --release
```
The executable is produced at `target/release/porcupine`.

### 3.2 Run Automated Tests
```bash
cargo test
cargo test --release
```

### 3.3 Run the Benchmark Suite
```bash
cargo run --release --bin nnue_bench
```

---

## 4. Training, Export, and Verification

### 4.1 Verify Model and Feature Parity
```bash
python3 training/features.py
python3 training/verify.py
```

### 4.2 Train a Model
```bash
python3 training/train.py --arch A3 --epochs 5 --batch-size 64 --output default.nnue
```

### 4.3 Run Architecture and Data-Efficiency Experiments
```bash
python3 training/run_experiments.py
```

---

## 5. UCI Configuration

The engine exposes standard UCI options for NNUE evaluation:

- `UseNNUE`: `check` (default `true`). Toggles between FH-NNUE and Hand-Crafted Evaluation (HCE) at runtime.
- `EvalFile`: `string` (default `default.nnue`). Path to the versioned `.nnue` model file.

Example UCI session:
```text
uci
isready
setoption name UseNNUE value true
setoption name EvalFile value default.nnue
position startpos
go depth 10
```
