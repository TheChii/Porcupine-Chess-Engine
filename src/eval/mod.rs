//! Board evaluation module.
//!
//! Uses NNUE if available, otherwise falls back to material.
//! Automatically switches to endgame evaluation when few pieces remain.

use crate::types::{Board, Score, Color, Piece, piece_value, Value, Move};

pub mod nnue;
pub mod hce;
pub mod endgame;

// Re-export the evaluator for use in search
pub use nnue::NnueEvaluator;
pub use endgame::{USE_ENDGAME_EVAL, should_use_endgame};

/// Evaluator wrapper that handles NNUE, HCE, Endgame, or Material evaluation
#[derive(Clone)]
pub enum SearchEvaluator<'a> {
    Nnue(NnueEvaluator<'a>),
    Hce,
    Endgame,
    Material,
}

impl<'a> SearchEvaluator<'a> {
    pub fn new(model: Option<&'a nnue::Model>, board: &Board) -> Self {
        match model {
            Some(m) => Self::Nnue(NnueEvaluator::new(&m.model, board)),
            None => Self::Hce, // Use HCE as fallback
        }
    }

    #[inline]
    pub fn evaluate(&mut self, board: &Board) -> Score {
        // Check if we should switch to endgame eval
        // This happens automatically when few pieces remain
        if USE_ENDGAME_EVAL && should_use_endgame(board) {
            return endgame::evaluate(board);
        }
        
        match self {
            Self::Nnue(e) => e.evaluate(board.turn()),
            Self::Hce => hce::evaluate(board),
            Self::Endgame => endgame::evaluate(board),
            Self::Material => material_eval_wrapper(board),
        }
    }

    #[inline]
    pub fn update_move(&mut self, board: &Board, m: Move) -> bool {
        match self {
            Self::Nnue(e) => e.update_move(board, m),
            Self::Hce => true,      // HCE is stateless
            Self::Endgame => true,  // Endgame is stateless
            Self::Material => true, // Material is stateless
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
/// Automatically switches to endgame eval when appropriate.
pub fn evaluate(board: &Board, model: Option<&nnue::Model>) -> Score {
    // Auto-switch to endgame eval
    if USE_ENDGAME_EVAL && should_use_endgame(board) {
        return endgame::evaluate(board);
    }
    
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
    fn test_endgame_auto_switch() {
        // KRK endgame should trigger endgame eval
        let board = Board::from_fen("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1").unwrap();
        assert!(should_use_endgame(&board));
    }
    
    #[test]
    fn test_opening_no_switch() {
        // Starting position should NOT use endgame eval
        let board = Board::default();
        assert!(!should_use_endgame(&board));
    }
}
