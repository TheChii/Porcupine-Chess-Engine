//! Board evaluation module.
//!
//! Uses NNUE if available, otherwise falls back to optimized HCE.
//! The HCE handles all game phases with tapered evaluation.

use crate::types::{Board, Score, Color, Piece, piece_value, Value, Move};

pub mod nnue;
pub mod hce;

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
            Some(m) => Self::Nnue(NnueEvaluator::new(&**m, board)),
            None => Self::Hce,
        }
    }

    #[inline]
    pub fn evaluate(&mut self, board: &Board) -> Score {
        match self {
            Self::Nnue(e) => e.evaluate(board.turn()),
            Self::Hce => hce::evaluate(board),
        }
    }

    #[inline]
    pub fn update_move(&mut self, board: &Board, m: Move) -> bool {
        match self {
            Self::Nnue(e) => e.update_move(board, m),
            Self::Hce => true, // HCE is stateless
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
/// Uses NNUE if a model is provided, otherwise HCE fallback.
pub fn evaluate(board: &Board, model: Option<&nnue::Model>) -> Score {
    if let Some(m) = model {
        // Use NNUE evaluation
        nnue::evaluate_scratch(&**m, board)
    } else {
        // Fallback to HCE
        hce::evaluate(board)
    }
}

/// Wrapper for material eval that returns Score
pub fn material_eval_wrapper(board: &Board) -> Score {
    let eval = material_eval(board);
    if board.turn() == Color::White {
        Score::cp(eval)
    } else {
        Score::cp(-eval)
    }
}

/// Simple material evaluation (white's perspective)
fn material_eval(board: &Board) -> Value {
    let mut score: Value = 0;

    for piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        let white_pieces = board.piece_bb(*piece) & board.color_bb(Color::White);
        let black_pieces = board.piece_bb(*piece) & board.color_bb(Color::Black);

        let white_count = white_pieces.count() as Value;
        let black_count = black_pieces.count() as Value;

        score += piece_value(*piece) * (white_count - black_count);
    }

    score
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
