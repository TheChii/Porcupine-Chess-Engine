//! Static Exchange Evaluation (SEE)
//!
//! Determines if a capture sequence is winning, losing, or neutral.
//! Uses fixed-size arrays to avoid allocations.

use crate::types::{Board, Move, Piece, Color, Bitboard};
use movegen::attacks::{pawn_attacks, knight_attacks, king_attacks, bishop_attacks, rook_attacks};

/// Piece values for SEE (using lower values for faster cutoffs)
const SEE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 20000]; // P, N, B, R, Q, K

/// Get SEE value for a piece
#[inline]
fn see_piece_value(piece: Piece) -> i32 {
    SEE_VALUES[piece.index()]
}

/// Get least valuable attacker of a square
#[inline]
fn get_lva(board: &Board, sq: movegen::Square, side: Color, occupied: Bitboard) -> Option<(movegen::Square, Piece)> {
    // Check each piece type from least to most valuable
    for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        let attackers = get_piece_attacks(board, sq, piece, side, occupied);
        if attackers.any() {
            return Some((unsafe { attackers.lsb_unchecked() }, piece));
        }
    }
    None
}

/// Get attacks from a specific piece type to a square
#[inline]
fn get_piece_attacks(board: &Board, target: movegen::Square, piece: Piece, side: Color, occupied: Bitboard) -> Bitboard {
    let our_pieces = board.piece_bb(piece) & board.color_bb(side) & occupied;
    
    match piece {
        Piece::Pawn => {
            // For pawn attacks TO a square, we need attacks FROM the opposite color
            let enemy_color = !side;
            let attacks_to_target = pawn_attacks(enemy_color, target);
            our_pieces & attacks_to_target
        }
        Piece::Knight => {
            our_pieces & knight_attacks(target)
        }
        Piece::Bishop => {
            our_pieces & bishop_attacks(target, occupied)
        }
        Piece::Rook => {
            our_pieces & rook_attacks(target, occupied)
        }
        Piece::Queen => {
            our_pieces & (bishop_attacks(target, occupied) | rook_attacks(target, occupied))
        }
        Piece::King => {
            our_pieces & king_attacks(target)
        }
    }
}

/// Static Exchange Evaluation
/// Returns the material balance after a capture sequence.
/// Uses fixed-size array to avoid allocations.
#[inline]
/// Static Exchange Evaluation with known victim
/// Returns the material balance after a capture sequence.
/// `victim` should be the piece at the target square (None for En Passant).
#[inline]
pub fn see_captured(board: &Board, mv: Move, victim: Option<Piece>) -> i32 {
    let from = mv.from();
    let to = mv.to();
    
    let attacker = board.piece_at(from).map(|(p, _)| p);
    
    let (attacker_piece, mut gain) = match (attacker, victim) {
        (Some(a), Some(v)) => (a, see_piece_value(v)),
        (Some(a), None) => {
            // En passant capture or quiet move?
            // SEE is typically only called for captures.
            // If it's a pawn moving to empty square, assume EP if it's a capture.
            if a == Piece::Pawn {
                (a, see_piece_value(Piece::Pawn))
            } else {
                return 0; // Not a capture
            }
        }
        _ => return 0,
    };

    // Handle promotion
    if let Some(promo) = mv.flag().promotion_piece() {
        gain += see_piece_value(promo) - see_piece_value(Piece::Pawn);
    }

    // Fixed-size gains stack (max 32 captures possible)
    let mut gains: [i32; 32] = [0; 32];
    let mut depth = 0;
    gains[depth] = gain;
    depth += 1;

    let mut occupied = board.occupied() ^ Bitboard::from_square(from);
    let mut side = !board.turn();
    let mut last_value = see_piece_value(attacker_piece);
    
    // Simulate the exchange
    loop {
        if let Some((sq, piece)) = get_lva(board, to, side, occupied) {
            occupied = occupied ^ Bitboard::from_square(sq);
            gains[depth] = last_value;
            last_value = see_piece_value(piece);
            depth += 1;
            side = !side;
            
            // King capture ends the sequence
            if piece == Piece::King {
                break;
            }
        } else {
            break;
        }
    }
    
    // Negamax-style evaluation from the end
    while depth > 1 {
        depth -= 1;
        gains[depth - 1] = gains[depth - 1].max(-gains[depth]);
    }
    
    gains[0]
}

/// Static Exchange Evaluation
/// Returns the material balance after a capture sequence.
#[inline]
pub fn see(board: &Board, mv: Move) -> i32 {
    let victim = board.piece_at(mv.to()).map(|(p, _)| p);
    see_captured(board, mv, victim)
}

/// Check if SEE is greater than or equal to threshold
#[inline]
pub fn see_ge(board: &Board, mv: Move, threshold: i32) -> bool {
    see(board, mv) >= threshold
}

/// Check if a capture is winning (SEE >= 0)
#[inline]
pub fn is_good_capture(board: &Board, mv: Move) -> bool {
    see_ge(board, mv, 0)
}

/// Check if a capture is winning (SEE >= 0) with known victim
#[inline]
pub fn is_good_capture_with_victim(board: &Board, mv: Move, victim: Option<Piece>) -> bool {
    see_captured(board, mv, victim) >= 0
}
