//! Lightweight Endgame HCE
//! Used only for positions with very few pieces (total pieces <= 6)
//! Optimised for speed and endgame accuracy.

use crate::types::{piece_value, Board, Color, Piece, Score, Value};

/// Simple centre distance table for king centralisation
const KING_CENTER_MAB: [Value; 64] = [
    -30, -20, -10, -10, -10, -10, -20, -30,
    -20, -10,   0,   0,   0,   0, -10, -20,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   0,  10,  20,  20,  10,   0, -10,
    -10,   0,  10,  20,  20,  10,   0, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -20, -10,   0,   0,   0,   0, -10, -20,
    -30, -20, -10, -10, -10, -10, -20, -30,
];

// ------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------

/// Chebyshev distance between two squares (king moves)
#[inline(always)]
fn king_dist(a: usize, b: usize) -> u32 {
    let fa = (a & 7) as i32;
    let ra = (a >> 3) as i32;
    let fb = (b & 7) as i32;
    let rb = (b >> 3) as i32;
    ((fa - fb).abs()).max((ra - rb).abs()) as u32
}

/// True if the two squares are adjacent (king move apart)
#[inline(always)]
fn are_adjacent(a: usize, b: usize) -> bool {
    king_dist(a, b) == 1
}

/// Key squares for a pawn of `pawn_color` standing on `sq`.
/// Returns the squares the attacking king must occupy to force promotion.
fn get_key_squares(sq: usize, pawn_color: Color) -> Vec<usize> {
    let file = sq & 7;
    let rank = sq >> 3;
    match pawn_color {
        Color::White => {
            if file == 0 {
                return vec![49, 57]; // b7, b8
            }
            if file == 7 {
                return vec![54, 62]; // g7, g8
            }
            if rank == 6 {
                // one step before promotion: key squares = promotion squares
                let promo_rank = 7;
                let mut sqs = Vec::new();
                if file > 0 { sqs.push((promo_rank << 3) | (file - 1)); }
                sqs.push((promo_rank << 3) | file);
                if file < 7 { sqs.push((promo_rank << 3) | (file + 1)); }
                return sqs;
            } else {
                let target_rank = rank + 2;
                let mut sqs = Vec::new();
                if file > 0 { sqs.push((target_rank << 3) | (file - 1)); }
                sqs.push((target_rank << 3) | file);
                if file < 7 { sqs.push((target_rank << 3) | (file + 1)); }
                return sqs;
            }
        }
        Color::Black => {
            if file == 0 {
                return vec![9, 1];  // b2, b1
            }
            if file == 7 {
                return vec![14, 6]; // g2, g1
            }
            if rank == 1 {
                // one step before promotion
                let promo_rank = 0;
                let mut sqs = Vec::new();
                if file > 0 { sqs.push((promo_rank << 3) | (file - 1)); }
                sqs.push((promo_rank << 3) | file);
                if file < 7 { sqs.push((promo_rank << 3) | (file + 1)); }
                return sqs;
            } else {
                let target_rank = rank - 2;
                let mut sqs = Vec::new();
                if file > 0 { sqs.push((target_rank << 3) | (file - 1)); }
                sqs.push((target_rank << 3) | file);
                if file < 7 { sqs.push((target_rank << 3) | (file + 1)); }
                return sqs;
            }
        }
    }
}

/// Is KPK a win for the side with the pawn?
/// `attacker_king` and `defender_king` are squares of the side with and without the pawn.
fn is_kpk_win(
    pawn_color: Color,
    attacker_king: usize,
    defender_king: usize,
    pawn_sq: usize,
    side_to_move: Color,
) -> bool {
    let pawn_to_move = side_to_move == pawn_color;

    // Immediate capture: if it's the defender's turn and he can safely take the pawn → draw
    if !pawn_to_move && are_adjacent(defender_king, pawn_sq) && !are_adjacent(attacker_king, pawn_sq) {
        return false;
    }

    // 1. Attacker already on a key square → win
    let key_squares = get_key_squares(pawn_sq, pawn_color);
    for &ksq in &key_squares {
        if attacker_king == ksq {
            return true;
        }
    }

    // 2. Race to a key square
    for &ksq in &key_squares {
        let w_dist = king_dist(attacker_king, ksq);
        let b_dist = king_dist(defender_king, ksq);
        if pawn_to_move {
            if w_dist <= b_dist { return true; }
        } else {
            if w_dist < b_dist { return true; }
        }
    }

    // 3. Square of the pawn – if the defender is outside, pawn runs
    let promo_sq = if pawn_color == Color::White {
        (7 << 3) | (pawn_sq & 7)
    } else {
        (0 << 3) | (pawn_sq & 7)
    };
    let promo_moves = if pawn_color == Color::White {
        7 - (pawn_sq >> 3)
    } else {
        pawn_sq >> 3
    };
    let effective_moves = if pawn_to_move { promo_moves } else { promo_moves + 1 };
    if king_dist(defender_king, promo_sq) > effective_moves as u32 {
        return true;
    }

    false
}

/// Check whether `color` has insufficient material to force checkmate.
/// Must be called when the opponent has only a bare king (no pawns/pieces).
fn is_insufficient_material_side(
    color: Color,
    knights: Value,
    bishops: Value,
    rooks: Value,
    queens: Value,
    board: &Board,
) -> bool {
    if rooks > 0 || queens > 0 { return false; }
    if knights == 0 && bishops == 0 { return true; } // bare king
    if knights == 1 && bishops == 0 { return true; }
    if knights == 2 && bishops == 0 { return true; } // 2N vs K is draw
    if knights == 0 && bishops == 1 { return true; }
    if knights == 0 && bishops == 2 {
        // 2B vs K: only a win if bishops are on opposite colours
        let bb = board.piece_bb(Piece::Bishop) & board.color_bb(color);
        let squares: Vec<usize> = bb.iter().map(|s| s.index() as usize).collect();
        if squares.len() == 2 {
            let sq1 = squares[0];
            let sq2 = squares[1];
            let dark1 = ((sq1 & 7) + (sq1 >> 3)) & 1 == 0;
            let dark2 = ((sq2 & 7) + (sq2 >> 3)) & 1 == 0;
            return dark1 == dark2; // same colour → draw
        }
    }
    false // e.g. N+B, 2B opposite colours, etc. → sufficient
}

// ------------------------------------------------------------------------
// Main evaluation entry point
// ------------------------------------------------------------------------

pub fn evaluate(board: &Board) -> Score {
    // -- Piece counts ----------------------------------------------------
    let w_pawns_bb = board.piece_bb(Piece::Pawn) & board.color_bb(Color::White);
    let b_pawns_bb = board.piece_bb(Piece::Pawn) & board.color_bb(Color::Black);
    let w_pawns = w_pawns_bb.count() as Value;
    let b_pawns = b_pawns_bb.count() as Value;

    let w_knights = (board.piece_bb(Piece::Knight) & board.color_bb(Color::White)).count() as Value;
    let b_knights = (board.piece_bb(Piece::Knight) & board.color_bb(Color::Black)).count() as Value;
    let w_bishops = (board.piece_bb(Piece::Bishop) & board.color_bb(Color::White)).count() as Value;
    let b_bishops = (board.piece_bb(Piece::Bishop) & board.color_bb(Color::Black)).count() as Value;
    let w_rooks   = (board.piece_bb(Piece::Rook) & board.color_bb(Color::White)).count() as Value;
    let b_rooks   = (board.piece_bb(Piece::Rook) & board.color_bb(Color::Black)).count() as Value;
    let w_queens  = (board.piece_bb(Piece::Queen) & board.color_bb(Color::White)).count() as Value;
    let b_queens  = (board.piece_bb(Piece::Queen) & board.color_bb(Color::Black)).count() as Value;

    let w_king_sq = board.king_square(Color::White).index() as usize;
    let b_king_sq = board.king_square(Color::Black).index() as usize;

    // "only king" flags
    let w_only_king = w_pawns == 0 && w_knights == 0 && w_bishops == 0 && w_rooks == 0 && w_queens == 0;
    let b_only_king = b_pawns == 0 && b_knights == 0 && b_bishops == 0 && b_rooks == 0 && b_queens == 0;

    // -- Both bare kings → draw -------------------------------------------
    if w_only_king && b_only_king {
        return Score::cp(0);
    }

    // -- KPK (one pawn, all other pieces missing) --------------------------
    if w_pawns == 1 && b_pawns == 0 && w_only_king && b_only_king {
        let pawn_sq = w_pawns_bb.lsb().unwrap().index() as usize;
        let win = is_kpk_win(Color::White, w_king_sq, b_king_sq, pawn_sq, board.turn());
        let score = if win { 700 } else { 0 };
        return Score::cp(if board.turn() == Color::White { score } else { -score });
    }
    if b_pawns == 1 && w_pawns == 0 && b_only_king && w_only_king {
        let pawn_sq = b_pawns_bb.lsb().unwrap().index() as usize;
        let win = is_kpk_win(Color::Black, b_king_sq, w_king_sq, pawn_sq, board.turn());
        let score = if win { -700 } else { 0 };
        return Score::cp(if board.turn() == Color::White { score } else { -score });
    }

    // -- Material difference (used in all remaining branches) --------------
    let material_score: Value = piece_value(Piece::Pawn)   * (w_pawns - b_pawns)
        + piece_value(Piece::Knight) * (w_knights - b_knights)
        + piece_value(Piece::Bishop) * (w_bishops - b_bishops)
        + piece_value(Piece::Rook)   * (w_rooks - b_rooks)
        + piece_value(Piece::Queen)  * (w_queens - b_queens);

    // -- Bare king + pieces (no pawns) ------------------------------------
    if w_only_king && !b_only_king {
        if is_insufficient_material_side(Color::Black, b_knights, b_bishops, b_rooks, b_queens, board) {
            return Score::cp(0);
        }
        let mut score = material_score - 300; // black's winning advantage
        score += KING_CENTER_MAB[w_king_sq] - KING_CENTER_MAB[b_king_sq];
        return Score::cp(if board.turn() == Color::White { score } else { -score });
    }
    if b_only_king && !w_only_king {
        if is_insufficient_material_side(Color::White, w_knights, w_bishops, w_rooks, w_queens, board) {
            return Score::cp(0);
        }
        let mut score = material_score + 300; // white's winning advantage
        score += KING_CENTER_MAB[w_king_sq] - KING_CENTER_MAB[b_king_sq];
        return Score::cp(if board.turn() == Color::White { score } else { -score });
    }

    // -- General evaluation for remaining positions (both sides have material) -
    let mut score = material_score;
    score += KING_CENTER_MAB[w_king_sq] - KING_CENTER_MAB[b_king_sq];

    // Pawn advance and passed pawn bonuses
    for sq in w_pawns_bb.iter() {
        let sq_idx = sq.index() as usize;
        let rank = (sq_idx >> 3) as Value;
        let file = (sq_idx & 7) as i32;
        score += rank * 5; // advance bonus
        let mut passed = true;
        for bsq in b_pawns_bb.iter() {
            let bsq_idx = bsq.index() as usize;
            let bfile = (bsq_idx & 7) as i32;
            let brank = (bsq_idx >> 3) as Value;
            if (bfile - file).abs() <= 1 && brank > rank {
                passed = false;
                break;
            }
        }
        if passed {
            score += 30;
            if rank == 7 { score += 50; }
        }
    }

    for sq in b_pawns_bb.iter() {
        let sq_idx = sq.index() as usize;
        let rank = (sq_idx >> 3) as Value;
        let file = (sq_idx & 7) as i32;
        let black_rank = 7 - rank;
        score -= black_rank * 5;
        let mut passed = true;
        for wsq in w_pawns_bb.iter() {
            let wsq_idx = wsq.index() as usize;
            let wfile = (wsq_idx & 7) as i32;
            let wrank = (wsq_idx >> 3) as Value;
            if (wfile - file).abs() <= 1 && wrank < rank {
                passed = false;
                break;
            }
        }
        if passed {
            score -= 30;
            if rank == 0 { score -= 50; }
        }
    }

    Score::cp(if board.turn() == Color::White { score } else { -score })
}