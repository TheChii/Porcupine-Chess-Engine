//! Board evaluation module.
//!
//! Uses NNUE if available, otherwise falls back to optimized HCE.
//! The HCE handles all game phases with tapered evaluation.

use crate::types::{piece_value, Board, Color, Move, Piece, Score, Value};

pub mod hce;
pub mod nnue;
pub mod endgame_hce;
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
        method: EvalMethod,
        model: Option<&'a nnue::Model>, 
        porcupine: Option<&porcupine_nnue::Model>, 
        board: &Board
    ) -> Self {
        match method {
            EvalMethod::Porcupine => {
                if let Some(m) = porcupine {
                    Self::Porcupine(PorcupineEvaluator::new(std::sync::Arc::new(m.clone()), board))
                } else {
                    Self::Hce
                }
            }
            EvalMethod::Nnue => {
                if let Some(m) = model {
                    Self::Nnue(NnueEvaluator::new(m, board))
                } else {
                    Self::Hce
                }
            }
            EvalMethod::Hce => Self::Hce,
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

/// Evaluate the position.
///
/// Uses NNUE if a model is provided, otherwise HCE fallback.

/*
pub fn evaluate(board: &Board, model: Option<&nnue::Model>) -> Score {,
    hce::evaluate(board);
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
*/
/// Evaluate the position using only HCE.
pub fn evaluate(board: &Board, _model: Option<&nnue::Model>) -> Score {
    // Force fallback to HCE regardless of whether a model is provided
    hce::evaluate(board)
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

    #[test]
    fn test_hce_fallback() {
        // Evaluate without NNUE should use HCE
        let board = Board::default();
        let score = evaluate(&board, None);
        assert!(score.raw().abs() < 50);
    }
}
