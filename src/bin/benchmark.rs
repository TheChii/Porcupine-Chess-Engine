//! Simple NNUE benchmark - run with: cargo run --release --bin benchmark

use std::time::Instant;
use chessinrust::types::Board;
use chessinrust::eval::nnue;

fn main() {
    let model = match nnue::load_model("network.nnue") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load network.nnue: {}", e);
            return;
        }
    };

    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/ppp1bppp/2n2n2/3pp3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 7",
        "r2qr1k1/1b1nbppp/p1p2n2/1p1pp3/4P3/1BNP1N1P/PPPB1PP1/R2QR1K1 w - - 0 12",
    ];

    const ITERS: u32 = 100_000;
    let mut total_ns = 0u128;

    for fen in positions {
        let board = Board::from_fen(fen).unwrap();
        let start = Instant::now();
        for _ in 0..ITERS {
            let mut state = nnue::create_state(&model.model, &board);
            std::hint::black_box(nnue::evaluate_state(&mut state, board.turn()));
        }
        total_ns += start.elapsed().as_nanos();
    }

    let total_evals = ITERS as u64 * positions.len() as u64;
    let evals_per_sec = (total_evals as f64 / (total_ns as f64 / 1_000_000_000.0)) as u64;
    println!("Total: {} evals, {} evals/sec", total_evals, evals_per_sec);
}
