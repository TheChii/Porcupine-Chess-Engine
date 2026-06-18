//! Board evaluation module.
//!
//! Uses optimized HCE.
//! The HCE handles all game phases with tapered evaluation.

use crate::types::{piece_value, Board, Color, Piece, Score, Value};

pub mod hce;

/// Evaluate the position using HCE.
pub fn evaluate(board: &Board) -> Score {
    hce::evaluate(board)
}

pub fn material_eval_wrapper(b: &Board) -> Score {
    let e = material_eval(b);
    if b.turn() == Color::White { Score::cp(e) } else { Score::cp(-e) }
}

fn material_eval(b: &Board) -> Value {
    let mut s: Value = 0;
    for p in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let w = (b.piece_bb(*p) & b.color_bb(Color::White)).count() as Value;
        let bl = (b.piece_bb(*p) & b.color_bb(Color::Black)).count() as Value;
        s += piece_value(*p) * (w - bl);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position_material() {
        let board = Board::default();
        let score = material_eval_wrapper(&board);
        assert!(score.raw().abs() < 50);
    }

    #[test]
    fn test_hce() {
        let board = Board::default();
        let score = evaluate(&board);
        assert!(score.raw().abs() < 50);
    }
}
