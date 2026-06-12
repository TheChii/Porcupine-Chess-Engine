//! Lightweight Endgame HCE
//! Used only for positions with very few pieces (e.g., total pieces <= 6)

use crate::types::{piece_value, Board, Color, Piece, Score, Value};

/// Simple center distance table for king centralization
const KING_CENTER_MAB: [Value; 64] = [
    -30, -20, -10, -10, -10, -10, -20, -30,
    -20, -10,   0,   0,   0,   0, -10, -20,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   0,  10,  20,  20,  10,   0, -10,
    -10,   0,  10,  20,  20,  10,   0, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -20, -10,   0,   0,   0,   0, -10, -20,
    -30, -20, -10, -10, -10, -10, -20, -30,
];

pub fn evaluate(board: &Board) -> Score {
    let mut score: Value = 0;

    // Material
    for piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let w_count = (board.piece_bb(*piece) & board.color_bb(Color::White)).count() as Value;
        let b_count = (board.piece_bb(*piece) & board.color_bb(Color::Black)).count() as Value;
        score += piece_value(*piece) * (w_count - b_count);
    }

    // King centralization
    let w_king = board.king_square(Color::White);
    let b_king = board.king_square(Color::Black);
    
    score += KING_CENTER_MAB[w_king.index() as usize];
    score -= KING_CENTER_MAB[b_king.index() as usize];

    // Convert to side-to-move score
    if board.turn() == Color::White {
        Score::cp(score)
    } else {
        Score::cp(-score)
    }
}
