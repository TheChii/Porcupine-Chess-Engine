<p align="center">
  <img src="icon.png" alt="Porcupine Chess Engine" width="200">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Rust-orange?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/UCI-Compatible-green?style=for-the-badge" alt="UCI">
  <img src="https://img.shields.io/badge/NNUE-HalfKP-purple?style=for-the-badge" alt="NNUE">
</p>

# 🦔 Porcupine Chess Engine

**Porcupine** is a high-performance, UCI-compatible chess engine written entirely in Rust. Combining cutting-edge search algorithms with NNUE (Efficiently Updatable Neural Network) evaluation, Porcupine delivers strong, tactical play while maintaining blazing-fast performance.

> *Porcupine is the successor of [Ferrum](https://github.com/TheChii/Ferrum), rebuilt with enhanced strength and new ideas.*

---

## ✨ Features

### 🧠 Neural Network Evaluation (NNUE)
- **HalfKP Architecture** — State-of-the-art `40960→256×2→32→32→1` neural network
- **Incremental Updates** — Efficient accumulator updates during search
- **Dual Perspective** — Separate accumulators for white and black views
- **Hand-Crafted Fallback** — Optimized HCE when NNUE unavailable

### 🔍 Advanced Search
| Technique | Description |
|-----------|-------------|
| **Alpha-Beta with PVS** | Principal Variation Search for optimal pruning |
| **Iterative Deepening** | Progressive depth with aspiration windows |
| **Lazy SMP** | Lock-free multi-threaded search |
| **Transposition Table** | Zobrist hashing with aging |
| **Quiescence Search** | Tactical resolution with SEE pruning |

### ✂️ Pruning & Reductions
- **Null Move Pruning** — Skip moves to prove beta cutoffs
- **Late Move Reductions (LMR)** — Reduced search for unlikely moves
- **Reverse Futility Pruning** — Early cutoffs with static margins
- **SEE Pruning** — Static Exchange Evaluation for captures
- **History Pruning** — Skip moves with poor historical performance
- **Futility Pruning** — Prune hopeless positions at low depths
- **ProbCut** — Probabilistic cutoffs based on shallow searches

### 📊 Move Ordering
1. **TT Move** — Best move from transposition table
2. **Good Captures** — MVV-LVA with SEE filtering
3. **Killer Moves** — Quiet moves that caused beta cutoffs
4. **Counter Moves** — Responses to opponent's previous move
5. **History Heuristic** — Butterfly and piece-to history tables

### ⚡ Performance
- **Custom Move Generator** — `ferrum-movegen` with magic bitboards
- **SIMD Optimizations** — Vectorized NNUE inference
- **Lock-Free TT** — Concurrent access without synchronization
- **Efficient Memory** — Minimal allocations in hot paths

---

## 🚀 Quick Start

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) 1.70+ (stable)
- Git

### Build from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/TheChii/Porcupine.git
cd Porcupine

# Build optimized release
cargo build --release

# Copy NNUE network to release folder
cp network.nnue target/release/
```

The executable will be at `target/release/porcupine` (or `porcupine.exe` on Windows).

### Download Pre-built
Check the [Releases](https://github.com/TheChii/Porcupine/releases) page for pre-compiled binaries.

---

## 🎮 Usage

Porcupine implements the **Universal Chess Interface (UCI)** protocol. Connect it to any UCI-compatible chess GUI:

| GUI | Platform | Link |
|-----|----------|------|
| **Arena** | Windows | [playwitharena.de](http://www.playwitharena.de/) |
| **CuteChess** | Cross-platform | [github.com/cutechess](https://github.com/cutechess/cutechess) |
| **En Croissant** | Cross-platform | [encroissant.org](https://www.encroissant.org/) |
| **Banksia GUI** | Cross-platform | [banksiagui.com](https://banksiagui.com/) |
| **Nibbler** | Cross-platform | [github.com/rooklift](https://github.com/rooklift/nibbler) |

### UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `Hash` | spin | 16 | Transposition table size (MB) |
| `Threads` | spin | 1 | Number of search threads |
| `MoveOverhead` | spin | 10 | Time buffer for communication (ms) |
| `OwnBook` | check | false | Use internal opening book |
| `BookPath` | string | — | Path to Polyglot opening book |

### Example Session

```
> uci
id name Porcupine
id author Chiriac Theodor
...
uciok

> isready
readyok

> position startpos moves e2e4 e7e5
> go depth 20
info depth 1 seldepth 1 score cp 35 nodes 21 nps 21000 time 1 pv g1f3
info depth 2 seldepth 4 score cp 28 nodes 89 nps 89000 time 1 pv g1f3 b8c6
...
bestmove g1f3

> quit
```

---

## 📁 Project Structure

```
Porcupine/
├── src/                    # Main engine source
│   ├── eval/               # Evaluation (NNUE + HCE)
│   ├── search/             # Search algorithm
│   │   ├── negamax.rs      # Main search loop
│   │   ├── qsearch.rs      # Quiescence search
│   │   ├── ordering.rs     # Move ordering
│   │   ├── tt.rs           # Transposition table
│   │   └── see.rs          # Static exchange evaluation
│   ├── uci/                # UCI protocol handler
│   ├── book/               # Opening book support
│   └── types/              # Core types (Board, Move, etc.)
├── ferrum-movegen/         # Move generation library
├── ferrum-nnue/            # NNUE inference library
├── network.nnue            # Default neural network
└── Cargo.toml              # Rust package manifest
```

---

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run perft validation
cargo test perft

# Benchmark NNUE inference
cargo test --release -p nnue benchmark
```

---

## 📈 Strength

Porcupine is designed to compete at a strong amateur level. Key factors contributing to its strength:

- ✅ Modern NNUE evaluation with HalfKP features
- ✅ Efficient search with proper pruning hierarchy
- ✅ Lazy SMP scaling on multi-core systems
- ✅ Solid time management with soft/hard limits

*Estimated strength: ~2200-2400 Elo (self-play testing)*

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- [ ] Improved evaluation tuning
- [ ] Additional pruning techniques
- [ ] Opening book generation
- [ ] Endgame tablebases support
- [ ] Cross-platform optimizations

Please open an issue to discuss major changes before submitting a PR.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Stockfish](https://stockfishchess.org/) — Inspiration for NNUE implementation
- [Chess Programming Wiki](https://www.chessprogramming.org/) — Invaluable resource
- [Bullet Trainer](https://github.com/jw1912/bullet) — NNUE training framework
- The Rust community for excellent tooling

---

<p align="center">
  <b>Made with ♟️ and 🦀</b>
</p>

---

## 📚 Keywords

*chess engine, rust chess, uci engine, nnue, neural network chess, alpha-beta search, chess ai, open source chess, chess programming, porcupine, romania*
