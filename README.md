# Porcupine Chess Engine

Porcupine is a UCI-compatible chess engine written in Rust. It utilizes a custom move generator and a HalfKP NNUE (Efficiently Updatable Neural Network) for evaluation.

## Build Instructions

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) 1.70+

### Building from Source
1. Clone the repository:
   ```bash
   git clone https://github.com/TheChii/Porcupine-Chess-Engine.git
   cd Porcupine-Chess-Engine
   ```

2. Build the optimized release binary:
   ```bash
   cargo build --release
   ```

The compiled executable will be located at `target/release/porcupine` (or `porcupine.exe` on Windows). Note that the default neural network is embedded directly into the binary during compilation.

## Usage

Porcupine implements the Universal Chess Interface (UCI) protocol and can be used with any UCI-compatible graphical user interface (GUI) or CLI tool (such as CuteChess, Arena, or Banksia GUI).

### Supported UCI Options
- `Hash`: Transposition table size in MB (Default: 16)
- `Threads`: Number of search threads (Default: 1)
- `MoveOverhead`: Time buffer for network communication in milliseconds (Default: 10)
- `OwnBook`: Use internal opening book (Default: false)
- `BookPath`: Path to Polyglot opening book

### Testing
To run the engine's test suite (including move generation and perft validations):
```bash
cargo test --release
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
