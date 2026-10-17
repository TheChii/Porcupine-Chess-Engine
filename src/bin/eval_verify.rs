use porcupine::types::Board;
use porcupine::nnue::model::NnueModel;
use porcupine::nnue::eval::evaluate_board;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "default.nnue".to_string());
    println!("Loading model: {}", model_path);
    let model = NnueModel::load_from_file(&model_path).expect("Failed to load model");
    let fens = [
        ("Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Startpos (Black)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
        ("After 1. e4 (Black)", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("White up Queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White down Queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
        ("Black up Queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"),
        ("Black down Queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
        ("White up Rook", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White down Rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w KQkq - 0 1"),
        ("Black up Rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b KQkq - 0 1"),
        ("Black down Rook", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
        ("Tactical 1", "r1bqk2r/pppp1ppp/2n5/1B2p3/1b2P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"),
        ("Game 5 pos", "r1b4k/pp1p1prp/1q3N1Q/2b2p2/4P3/P1P2p2/1P4PP/2KR1B1R w - - 0 20"),
        ("Game 5 after e4f5", "r1b4k/pp1p1prp/1q3N1Q/2b2P2/8/P1P2p2/1P4PP/2KR1B1R b - - 0 20"),
        ("Game 5 after Bxe3+", "r1b4k/pp1p1prp/1q3N1Q/5P2/8/P1P1bp2/1P4PP/2KR1B1R w - - 0 21"),
        ("Game 5 after Rd4", "r1b4k/pp1p1prp/1q3N1Q/2b2p2/3RP3/P1P2p2/1P4PP/2K2B1R b - - 1 20"),
        ("Game 5 after Bxd4", "r1b4k/pp1p1prp/1q3N1Q/5p2/3bP3/P1P2p2/1P4PP/2K2B1R w - - 0 21"),
        ("Game 5 after cxd4", "r1b4k/pp1p1prp/1q3N1Q/5p2/3PP3/P4p2/1P4PP/2K2B1R b - - 0 21"),
    ];

    println!("{:<18} | {:<12}", "Position", "Rust Score");
    println!("{:-<18}-+-{:-<12}", "", "");
    for (name, fen) in fens {
        let board = Board::from_fen(fen).expect("Invalid FEN");
        let score = evaluate_board(&board, &model);
        println!("{:<18} | {:>+6} cp", name, score.raw());
    }
}
