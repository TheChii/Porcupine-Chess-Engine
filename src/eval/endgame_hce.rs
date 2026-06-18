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

#[inline(always)]
fn king_dist(a: usize, b: usize) -> u32 {
    let (fa, ra) = ((a & 7) as i32, (a >> 3) as i32);
    let (fb, rb) = ((b & 7) as i32, (b >> 3) as i32);
    ((fa - fb).abs()).max((ra - rb).abs()) as u32
}

#[inline(always)]
fn are_adjacent(a: usize, b: usize) -> bool {
    king_dist(a, b) == 1
}

fn get_key_squares(sq: usize, pc: Color) -> Vec<usize> {
    let f = sq & 7;
    let r = sq >> 3;
    match pc {
        Color::White => {
            if f == 0 { return vec![49, 57]; }
            if f == 7 { return vec![54, 62]; }
            if r == 6 {
                let pr = 7;
                let mut sqs = Vec::new();
                if f > 0 { sqs.push((pr << 3) | (f - 1)); }
                sqs.push((pr << 3) | f);
                if f < 7 { sqs.push((pr << 3) | (f + 1)); }
                return sqs;
            } else {
                let tr = r + 2;
                let mut sqs = Vec::new();
                if f > 0 { sqs.push((tr << 3) | (f - 1)); }
                sqs.push((tr << 3) | f);
                if f < 7 { sqs.push((tr << 3) | (f + 1)); }
                return sqs;
            }
        }
        Color::Black => {
            if f == 0 { return vec![9, 1]; }
            if f == 7 { return vec![14, 6]; }
            if r == 1 {
                let pr = 0;
                let mut sqs = Vec::new();
                if f > 0 { sqs.push((pr << 3) | (f - 1)); }
                sqs.push((pr << 3) | f);
                if f < 7 { sqs.push((pr << 3) | (f + 1)); }
                return sqs;
            } else {
                let tr = r - 2;
                let mut sqs = Vec::new();
                if f > 0 { sqs.push((tr << 3) | (f - 1)); }
                sqs.push((tr << 3) | f);
                if f < 7 { sqs.push((tr << 3) | (f + 1)); }
                return sqs;
            }
        }
    }
}

fn is_kpk_win(pc: Color, ak: usize, dk: usize, psq: usize, stm: Color) -> bool {
    let ptm = stm == pc;
    if !ptm && are_adjacent(dk, psq) && !are_adjacent(ak, psq) { return false; }
    let ksqs = get_key_squares(psq, pc);
    for &ksq in &ksqs { if ak == ksq { return true; } }
    for &ksq in &ksqs {
        let wd = king_dist(ak, ksq);
        let bd = king_dist(dk, ksq);
        if if ptm { wd <= bd } else { wd < bd } { return true; }
    }
    let prs = if pc == Color::White { (7 << 3) | (psq & 7) } else { (0 << 3) | (psq & 7) };
    let pms = if pc == Color::White { 7 - (psq >> 3) } else { psq >> 3 };
    let ems = if ptm { pms } else { pms + 1 };
    if king_dist(dk, prs) > ems as u32 { return true; }
    false
}

fn is_insufficient_material_side(c: Color, n: Value, b: Value, r: Value, q: Value, brd: &Board) -> bool {
    if r > 0 || q > 0 { return false; }
    if n == 0 && b == 0 { return true; }
    if n == 1 && b == 0 { return true; }
    if n == 2 && b == 0 { return true; }
    if n == 0 && b == 1 { return true; }
    if n == 0 && b == 2 {
        let bb = brd.piece_bb(Piece::Bishop) & brd.color_bb(c);
        let sqs: Vec<usize> = bb.iter().map(|s| s.index() as usize).collect();
        if sqs.len() == 2 {
            let (sq1, sq2) = (sqs[0], sqs[1]);
            return (((sq1 & 7) + (sq1 >> 3)) & 1) == (((sq2 & 7) + (sq2 >> 3)) & 1);
        }
    }
    false
}

pub fn evaluate(board: &Board) -> Score {
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