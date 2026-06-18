//! Board evaluation module.
//!
//! Uses NNUE if available, otherwise falls back to optimized HCE.
//! The HCE handles all game phases with tapered evaluation.

use crate::types::{piece_value, Board, Color, Move, Piece, Score, Value};

pub mod hce;
pub mod nnue;
pub mod endgame_hce;

// Re-export the evaluator for use in search
pub use nnue::NnueEvaluator;

/// Evaluator wrapper that handles NNUE or HCE evaluation
#[derive(Clone)]
pub enum SearchEvaluator<'a> {
    Nnue(NnueEvaluator<'a>),
    Hce,
}

impl<'a> SearchEvaluator<'a> {
    pub fn new(model: Option<&'a nnue::Model>, board: &Board) -> Self {
        match model {
            Some(m) => Self::Nnue(NnueEvaluator::new(m, board)),
            None => Self::Hce,
        }
    }

    #[inline]
    pub fn evaluate(&mut self, ply: usize, board: &Board) -> Score {
        match self {
            Self::Nnue(e) => {
                if (board.color_bb(Color::White) | board.color_bb(Color::Black)).count() <= 6 {
                    endgame_hce::evaluate(board)
                } else {
                    e.evaluate(ply, board.turn())
                }
            }
            Self::Hce => hce::evaluate(board),
        }
    }

    #[inline]
    pub fn update_move(&mut self, ply: usize, board: &Board, m: Move) -> bool {
        match self {
            Self::Nnue(e) => e.update_move(ply, board, m),
            Self::Hce => true, // HCE is stateless
        }
    }

    #[inline]
    pub fn refresh(&mut self, ply: usize, board: &Board) {
        if let Self::Nnue(e) = self {
            e.refresh(ply, board);
        }
    }
}

/// Evaluate the position.
///
/// Uses NNUE if a model is provided, otherwise HCE fallback.
pub fn evaluate(board: &Board, model: Option<&nnue::Model>) -> Score {
    if let Some(m) = model {
        if (board.color_bb(Color::White) | board.color_bb(Color::Black)).count() <= 6 {
            endgame_hce::evaluate(board)
        } else {
            nnue::evaluate_scratch(m, board)
        }
    } else {
        // Fallback to HCE
        hce::evaluate(board)
    }
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
    fn test_hce_fallback() {
        // Evaluate without NNUE should use HCE
        let board = Board::default();
        let score = evaluate(&board, None);
        assert!(score.raw().abs() < 50);
    }
}
