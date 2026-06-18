//! Board evaluation module.
//!
//! Uses NNUE if available, otherwise falls back to optimized HCE.
//! The HCE handles all game phases with tapered evaluation.

use crate::types::{piece_value, Board, Color, Move, Piece, Score, Value};

pub mod hce;
pub mod nnue;
pub mod porcupine_nnue;

// Re-export the evaluator for use in search
pub use nnue::NnueEvaluator;
pub use porcupine_nnue::PorcupineEvaluator;

/// Evaluation methods available in the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMethod {
    Hce,
    Nnue,      // Default HalfKP
    Porcupine, // Custom 768->128->1
}

/// Evaluator wrapper that handles NNUE or HCE evaluation
#[derive(Clone)]
pub enum SearchEvaluator<'a> {
    Nnue(NnueEvaluator<'a>),
    Porcupine(PorcupineEvaluator),
    Hce,
}

impl<'a> SearchEvaluator<'a> {
    pub fn new(
        _method: EvalMethod, // Ignored, kept for compatibility
        _model: Option<&'a nnue::Model>, 
        _porcupine: Option<&porcupine_nnue::Model>, 
        _board: &Board
    ) -> Self {
        // ==========================================
        //         EVALUATION METHOD SWITCH
        // ==========================================
        // Comment/uncomment exactly ONE of the 'return' lines below
        // to change the engine's evaluation method:

        // 1. Porcupine Custom NNUE (768->128->1)
        //return if let Some(m) = _porcupine { Self::Porcupine(PorcupineEvaluator::new(std::sync::Arc::new(m.clone()), _board)) } else { Self::Hce };

        // 2. Pure HCE (Hand-Crafted Evaluation)
        return Self::Hce;

        // 3. Default HalfKP NNUE (with HCE endgame fallback)
        // return if let Some(m) = _model { Self::Nnue(NnueEvaluator::new(m, _board)) } else { Self::Hce };
    }

    #[inline]
    pub fn evaluate(&mut self, ply: usize, board: &Board) -> Score {
        match self {
            Self::Nnue(e) => {
                // Default NNUE
                e.evaluate(ply, board.turn())
            }
            Self::Porcupine(e) => {
                // Use Porcupine NNUE
                e.evaluate(ply, board.turn())
            }
            Self::Hce => hce::evaluate(board),
        }
    }

    #[inline]
    pub fn update_move(&mut self, ply: usize, board: &Board, m: Move) -> bool {
        match self {
            Self::Nnue(e) => e.update_move(ply, board, m),
            Self::Porcupine(e) => e.update_move(ply, board, m),
            Self::Hce => true, // HCE is stateless
        }
    }

    #[inline]
    pub fn refresh(&mut self, ply: usize, board: &Board) {
        match self {
            Self::Nnue(e) => e.refresh(ply, board),
            Self::Porcupine(e) => e.refresh(ply, board),
            Self::Hce => (),
        }
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

    for piece in &[
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ] {
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


}
