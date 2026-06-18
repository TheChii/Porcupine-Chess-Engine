# Porcupine

Porcupine is a UCI compatible chess engine written in Rust. It uses a custom move generator and hand crafted evaluation.

## Build

1. Clone the repository
```bash
git clone https://github.com/TheChii/Porcupine-Chess-Engine.git
cd Porcupine-Chess-Engine
```

2. Build the release binary
```bash
cargo build --release
```

The executable will be located at target/release/porcupine.

## UCI Options

* Hash: Transposition table size in MB (Default: 16)
* Threads: Number of search threads (Default: 1)
* MoveOverhead: Time buffer in milliseconds (Default: 10)
* OwnBook: Use internal opening book (Default: false)
* BookPath: Path to Polyglot opening book

## Test

```bash
cargo test --release
```

## License

MIT License.
