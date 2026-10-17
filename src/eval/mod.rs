//! Board evaluation module.
//!
//! Uses optimized HCE.
//! The HCE handles all game phases with tapered evaluation.

use crate::types::{piece_value, Board, Color, Piece, Score, Value};
use std::sync::atomic::{AtomicU64, Ordering};

pub mod hce;

/// Diagnostic counters for auditing evaluation calls
pub static NNUE_EVAL_COUNT: AtomicU64 = AtomicU64::new(0);
pub static HCE_EVAL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset evaluation counters
pub fn reset_eval_counters() {
    NNUE_EVAL_COUNT.store(0, Ordering::Relaxed);
    HCE_EVAL_COUNT.store(0, Ordering::Relaxed);
}

/// Evaluate the position using NNUE (if enabled and loaded) or fallback to HCE.
#[inline]
pub fn evaluate(board: &Board) -> Score {
    if crate::nnue::is_nnue_enabled() {
        if let Some(model) = crate::nnue::get_global_model() {
            NNUE_EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
            return crate::nnue::evaluate_board(board, &model);
        }
    }
    HCE_EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
    hce::evaluate(board)
}

/// Fast evaluation using an incrementally maintained accumulator.
#[inline]
pub fn evaluate_with_accumulator(
    board: &Board,
    acc: &crate::nnue::PerspectiveAccumulators,
) -> Score {
    if crate::nnue::is_nnue_enabled() {
        if let Some(model) = crate::nnue::get_global_model() {
            NNUE_EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
            return crate::nnue::evaluate_from_accumulators(acc, board.turn(), &model);
        }
    }
    HCE_EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
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
