use porcupine::types::{Board, Color};
use porcupine::nnue::enumerate_features;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: feature_dump <fen>");
        std::process::exit(1);
    }
    let fen = &args[1];
    let board = Board::from_fen(fen).expect("Invalid FEN");

    let mut w_feats: Vec<usize> = enumerate_features(&board, Color::White).to_vec();
    let mut b_feats: Vec<usize> = enumerate_features(&board, Color::Black).to_vec();
    w_feats.sort_unstable();
    b_feats.sort_unstable();

    println!(
        "{{\"fen\":\"{}\",\"white\":{:?},\"black\":{:?}}}",
        fen, w_feats, b_feats
    );
}
