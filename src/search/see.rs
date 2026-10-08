//! Static Exchange Evaluation (SEE)
//!
//! Determines if a capture sequence is winning, losing, or neutral.
//! Uses fixed-size arrays to avoid allocations.

use crate::types::{Bitboard, Board, Color, Move, Piece};
use movegen::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};

/// Piece values for SEE (using lower values for faster cutoffs)
const SEE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 20000]; // P, N, B, R, Q, K

/// Get SEE value for a piece
#[inline]
fn see_piece_value(piece: Piece) -> i32 {
    SEE_VALUES[piece.index()]
}

#[inline]
fn get_lva(
    board: &Board,
    sq: movegen::Square,
    side: Color,
    occupied: Bitboard,
) -> Option<(movegen::Square, Piece)> {
    let side_bb = board.color_bb(side) & occupied;

    // 1. Pawns (least valuable)
    let enemy_color = !side;
    let pawn_attackers = board.piece_bb(Piece::Pawn) & side_bb & pawn_attacks(enemy_color, sq);
    if pawn_attackers.any() {
        return Some((unsafe { pawn_attackers.lsb_unchecked() }, Piece::Pawn));
    }

    // 2. Knights
    let knight_attackers = board.piece_bb(Piece::Knight) & side_bb & knight_attacks(sq);
    if knight_attackers.any() {
        return Some((unsafe { knight_attackers.lsb_unchecked() }, Piece::Knight));
    }

    // 3. Bishops (compute diagonal attacks once for bishops and queens)
    let diag_attacks = bishop_attacks(sq, occupied);
    let bishop_attackers = board.piece_bb(Piece::Bishop) & side_bb & diag_attacks;
    if bishop_attackers.any() {
        return Some((unsafe { bishop_attackers.lsb_unchecked() }, Piece::Bishop));
    }

    // 4. Rooks (compute straight attacks once for rooks and queens)
    let straight_attacks = rook_attacks(sq, occupied);
    let rook_attackers = board.piece_bb(Piece::Rook) & side_bb & straight_attacks;
    if rook_attackers.any() {
        return Some((unsafe { rook_attackers.lsb_unchecked() }, Piece::Rook));
    }

    // 5. Queens (reuse already computed diagonal and straight attacks)
    let queen_attackers = board.piece_bb(Piece::Queen) & side_bb & (diag_attacks | straight_attacks);
    if queen_attackers.any() {
        return Some((unsafe { queen_attackers.lsb_unchecked() }, Piece::Queen));
    }

    // 6. King
    let king_attackers = board.piece_bb(Piece::King) & side_bb & king_attacks(sq);
    if king_attackers.any() {
        return Some((unsafe { king_attackers.lsb_unchecked() }, Piece::King));
    }

    None
}

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
            if mv.flag() == movegen::MoveFlag::EnPassant {
                (a, see_piece_value(Piece::Pawn))
            } else {
                (a, 0)
            }
        }
        _ => return 0,
    };

    let mut last_value = see_piece_value(attacker_piece);

    // Handle promotion
    if let Some(promo) = mv.flag().promotion_piece() {
        gain += see_piece_value(promo) - see_piece_value(Piece::Pawn);
        last_value = see_piece_value(promo);
    }

    // Fixed-size gains stack (max 32 captures possible)
    let mut gains: [i32; 32] = [0; 32];
    let mut depth = 0;
    gains[depth] = gain;

    let mut occupied = board.occupied() ^ Bitboard::from_square(from);
    let mut side = !board.turn();

    // Simulate the exchange
    loop {
        depth += 1;
        gains[depth] = last_value;

        if depth >= 31 {
            break;
        }

        if let Some((sq, piece)) = get_lva(board, to, side, occupied) {
            occupied ^= Bitboard::from_square(sq);
            last_value = see_piece_value(piece);
            side = !side;

            // King capture ends the sequence
            if piece == Piece::King {
                gains[depth] = last_value;
                break;
            }
        } else {
            break;
        }
    }

    // Negamax-style evaluation from the end
    while depth > 1 {
        depth -= 1;
        gains[depth - 1] = gains[depth - 1] - gains[depth].max(0);
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

/// Fast check if SEE is greater than or equal to threshold with known victim
#[inline]
pub fn see_captured_ge(board: &Board, mv: Move, victim: Option<Piece>, threshold: i32) -> bool {
    let from = mv.from();

    let attacker = match board.piece_at(from) {
        Some((p, _)) => p,
        None => return false,
    };

    let mut gain = match victim {
        Some(v) => see_piece_value(v),
        None => {
            if mv.flag() == movegen::MoveFlag::EnPassant {
                see_piece_value(Piece::Pawn)
            } else {
                0
            }
        }
    };

    let mut piece_on_sq = attacker;
    if let Some(promo) = mv.flag().promotion_piece() {
        gain += see_piece_value(promo) - see_piece_value(Piece::Pawn);
        piece_on_sq = promo;
    }

    // 1. If best-case gain is less than threshold, we can never reach threshold
    if gain < threshold {
        return false;
    }

    // 2. If worst-case gain (even if opponent recaptures piece_on_sq) is >= threshold,
    // we are guaranteed >= threshold since we can stand pat
    if gain - see_piece_value(piece_on_sq) >= threshold {
        return true;
    }

    // 3. Fallback to full SEE
    see_captured(board, mv, victim) >= threshold
}

/// Check if SEE is greater than or equal to threshold
#[inline]
pub fn see_ge(board: &Board, mv: Move, threshold: i32) -> bool {
    let victim = board.piece_at(mv.to()).map(|(p, _)| p);
    see_captured_ge(board, mv, victim, threshold)
}

/// Check if a capture is winning (SEE >= 0)
#[inline]
pub fn is_good_capture(board: &Board, mv: Move) -> bool {
    see_ge(board, mv, 0)
}

/// Check if a capture is winning (SEE >= 0) with known victim
#[inline]
pub fn is_good_capture_with_victim(board: &Board, mv: Move, victim: Option<Piece>) -> bool {
    see_captured_ge(board, mv, victim, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_see_basic() {
        // Position where White pawn on e4 can take Black pawn on d5 protected by queen
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let e4_d5 = Move::new(movegen::Square::E4, movegen::Square::D5, movegen::MoveFlag::Capture);
        assert_eq!(see(&board, e4_d5), 0);
        assert!(see_ge(&board, e4_d5, 0));
        assert!(see_ge(&board, e4_d5, -50));
        assert!(!see_ge(&board, e4_d5, 50));
    }

    #[test]
    fn test_see_unprotected_piece() {
        // Black knight on e5 is completely hanging (no defenders)
        let board = Board::from_fen("8/8/8/4n3/4P3/8/8/4K2k w - - 0 1").unwrap();
        let e4_e5 = Move::new(movegen::Square::E4, movegen::Square::E5, movegen::MoveFlag::Capture);
        assert_eq!(see(&board, e4_e5), 300);
        assert!(see_ge(&board, e4_e5, 200));
        assert!(see_ge(&board, e4_e5, 300));
        assert!(!see_ge(&board, e4_e5, 301));
    }

    #[test]
    fn test_see_bad_capture() {
        // White Queen on d1 captures pawn on d5 protected by Black queen on d8
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let q_takes_d5 = Move::new(movegen::Square::D1, movegen::Square::D5, movegen::MoveFlag::Capture);
        assert_eq!(see(&board, q_takes_d5), 100 - 900);
        assert!(!see_ge(&board, q_takes_d5, 0));
        assert!(see_ge(&board, q_takes_d5, -800));
    }
}

