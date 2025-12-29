//! Hand-Crafted Evaluation (HCE) - Speed Optimized
//!
//! Ultra-fast classical evaluation with:
//! - Material + PST in single pass
//! - Tapered evaluation
//! - Minimal branching

use crate::types::{Board, Score, Color, Piece, Bitboard};
use movegen::Square;

// ============================================================================
// PIECE VALUES (packed: [MG, EG] per piece)
// ============================================================================

const PIECE_VAL: [(i32, i32); 6] = [
    (100, 120),   // Pawn
    (320, 300),   // Knight
    (330, 320),   // Bishop
    (500, 550),   // Rook
    (950, 1000),  // Queen
    (0, 0),       // King
];

// ============================================================================
// PIECE-SQUARE TABLES (packed MG, indexed by piece * 64 + square)
// Using flat array for cache efficiency
// ============================================================================

#[rustfmt::skip]
static PST_MG: [i32; 384] = [
    // Pawn (0-63)
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
    // Knight (64-127)
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
    // Bishop (128-191)
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
    // Rook (192-255)
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
    // Queen (256-319)
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
    // King MG (320-383)
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
];

#[rustfmt::skip]
static PST_EG: [i32; 384] = [
    // Pawn EG (0-63)
     0,  0,  0,  0,  0,  0,  0,  0,
   100,100,100,100,100,100,100,100,
    60, 60, 60, 60, 60, 60, 60, 60,
    40, 40, 40, 40, 40, 40, 40, 40,
    25, 25, 25, 25, 25, 25, 25, 25,
    10, 10, 10, 10, 10, 10, 10, 10,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
    // Knight EG (64-127) - same as MG
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
    // Bishop EG (128-191)
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
    // Rook EG (192-255)
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
    // Queen EG (256-319)
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
    // King EG (320-383) - centralization
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
];

const BISHOP_PAIR: i32 = 30;

// ============================================================================
// EVALUATION
// ============================================================================

/// Main evaluation function - returns score from side-to-move perspective
#[inline]
pub fn evaluate(board: &Board) -> Score {
    // Calculate phase once
    let phase = {
        let n = board.piece_bb(Piece::Knight).count() as i32;
        let b = board.piece_bb(Piece::Bishop).count() as i32;
        let r = board.piece_bb(Piece::Rook).count() as i32;
        let q = board.piece_bb(Piece::Queen).count() as i32;
        let material = n + b + 2 * r + 4 * q;
        ((24 - material) * 256 / 24).clamp(0, 256)
    };

    let mut mg = 0i32;
    let mut eg = 0i32;

    // Evaluate white
    eval_color::<true>(board, &mut mg, &mut eg);
    // Evaluate black (subtract)
    eval_color::<false>(board, &mut mg, &mut eg);

    // Taper
    let score = (mg * (256 - phase) + eg * phase) / 256;

    if board.turn() == Color::White {
        Score::cp(score)
    } else {
        Score::cp(-score)
    }
}

/// Evaluate one color using const generic for sign
#[inline(always)]
fn eval_color<const IS_WHITE: bool>(board: &Board, mg: &mut i32, eg: &mut i32) {
    let color = if IS_WHITE { Color::White } else { Color::Black };
    let sign = if IS_WHITE { 1 } else { -1 };

    // Process each piece type
    for (piece_idx, &piece) in [Piece::Pawn, Piece::Knight, Piece::Bishop, 
                                 Piece::Rook, Piece::Queen, Piece::King].iter().enumerate() {
        let pieces = board.piece_bb(piece) & board.color_bb(color);
        let base_idx = piece_idx * 64;
        let (mat_mg, mat_eg) = PIECE_VAL[piece_idx];

        for sq in pieces {
            let sq_idx = if IS_WHITE { (sq.index() as usize) ^ 56 } else { sq.index() as usize };
            let pst_idx = base_idx + sq_idx;

            // Material (not for king)
            if piece_idx < 5 {
                *mg += sign * mat_mg;
                *eg += sign * mat_eg;
            }

            // PST
            *mg += sign * PST_MG[pst_idx];
            *eg += sign * PST_EG[pst_idx];
        }
    }

    // Bishop pair
    if (board.piece_bb(Piece::Bishop) & board.color_bb(color)).count() >= 2 {
        *mg += sign * BISHOP_PAIR;
        *eg += sign * BISHOP_PAIR;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position() {
        let board = Board::default();
        let score = evaluate(&board);
        assert!(score.raw().abs() < 50);
    }

    #[test]
    fn test_material_advantage() {
        let board = Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score = evaluate(&board);
        assert!(score.raw() > 800);
    }
}
