//! Board evaluation module.
//!
//! Uses NNUE if available, otherwise falls back to material.

use crate::types::{Board, Score, Color, Piece, piece_value, Value};
// use crate::uci::UciHandler;

pub mod nnue;

// Re-export the evaluator for use in search
pub use nnue::NnueEvaluator;
use crate::types::Move;

/// Evaluator wrapper that handles either NNUE (incremental) or Material (stateless)
#[derive(Clone)]
pub enum SearchEvaluator<'a> {
    Nnue(NnueEvaluator<'a>),
    Material,
}

impl<'a> SearchEvaluator<'a> {
    pub fn new(model: Option<&'a nnue::Model>, board: &Board) -> Self {
        match model {
            Some(m) => Self::Nnue(NnueEvaluator::new(&m.model, board)),
            None => Self::Material,
        }
    }

    #[inline]
    pub fn evaluate(&mut self, board: &Board) -> Score {
        match self {
            Self::Nnue(e) => e.evaluate(board.side_to_move()),
            Self::Material => material_eval_wrapper(board),
        }
    }

    #[inline]
    pub fn update_move(&mut self, board: &Board, m: Move) -> bool {
        match self {
            Self::Nnue(e) => e.update_move(board, m),
            Self::Material => true, // Material eval is stateless, update always "succeeds"
        }
    }

    #[inline]
    pub fn refresh(&mut self, board: &Board) {
        if let Self::Nnue(e) = self {
            e.refresh(board);
        }
    }
}

/// Evaluate the position.
///
/// Uses NNUE if a model is provided, otherwise simple material fallback.
pub fn evaluate(board: &Board, model: Option<&nnue::Model>) -> Score {
    if let Some(m) = model {
        // Use NNUE evaluation
        nnue::evaluate_scratch(&m.model, board)
    } else {
        // Fallback to simple material
        material_eval_wrapper(board)
    }
}

/// Wrapper for material eval that returns Score
pub fn material_eval_wrapper(board: &Board) -> Score {
    let eval = material_eval(board);
    if board.side_to_move() == Color::White {
        Score::cp(eval)
    } else {
        Score::cp(-eval)
    }
}

/// Simple material evaluation (white's perspective)
fn material_eval(board: &Board) -> Value {
    let mut score: Value = 0;

    for piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let white_pieces = board.pieces(*piece) & board.color_combined(Color::White);
        let black_pieces = board.pieces(*piece) & board.color_combined(Color::Black);

        let white_count = white_pieces.popcnt() as Value;
        let black_count = black_pieces.popcnt() as Value;

        score += piece_value(*piece) * (white_count - black_count);
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_starting_position_material() {
        let board = Board::default();
        let score = material_eval_wrapper(&board);
        assert!(score.raw().abs() < 50);
    }
}
