//! Move ordering heuristics.
//!
//! Good move ordering is critical for alpha-beta pruning efficiency.
//! This module provides ordering functions with:
//! - Transposition table moves (best first)
//! - Captures via MVV-LVA
//! - Killer moves (quiet moves that caused cutoffs)
//! - Promotion bonuses

use crate::types::{Board, Move, Ply, piece_value};

/// Move score constants
const TT_MOVE_BONUS: i32 = 1_000_000;
const PROMOTION_BONUS: i32 = 100_000;
const CAPTURE_BONUS: i32 = 50_000;
const KILLER_0_BONUS: i32 = 40_000;
const KILLER_1_BONUS: i32 = 35_000;

/// MVV-LVA scores for capture ordering
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let victim = board.piece_on(m.get_dest());
    let attacker = board.piece_on(m.get_source());

    match (victim, attacker) {
        (Some(v), Some(a)) => {
            piece_value(v) * 10 - piece_value(a)
        }
        _ => 0,
    }
}

/// Score a move for ordering (higher = search first)
fn score_move(
    board: &Board, 
    m: Move, 
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
) -> i32 {
    // TT move is always searched first
    if tt_move == Some(m) {
        return TT_MOVE_BONUS;
    }

    let mut score = 0;

    // Promotions are very important
    if let Some(promo) = m.get_promotion() {
        score += piece_value(promo) + PROMOTION_BONUS;
    }

    // Captures scored by MVV-LVA
    if board.piece_on(m.get_dest()).is_some() {
        score += mvv_lva_score(board, m) + CAPTURE_BONUS;
    } else {
        // Quiet move - check killers
        if killers[0] == Some(m) {
            score += KILLER_0_BONUS;
        } else if killers[1] == Some(m) {
            score += KILLER_1_BONUS;
        }
    }

    score
}

/// Order moves for main search with TT move and killers priority
pub fn order_moves_with_tt_and_killers(
    board: &Board, 
    moves: &mut [Move], 
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
) {
    let mut scored: Vec<(Move, i32)> = moves.iter()
        .map(|&m| (m, score_move(board, m, tt_move, killers)))
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1));

    for (i, (m, _)) in scored.into_iter().enumerate() {
        moves[i] = m;
    }
}

/// Order moves for main search with TT move only (no killers)
pub fn order_moves_with_tt(board: &Board, moves: &mut [Move], tt_move: Option<Move>) {
    order_moves_with_tt_and_killers(board, moves, tt_move, [None, None]);
}

/// Order moves for main search (without TT move or killers)
#[allow(dead_code)]
pub fn order_moves(board: &Board, moves: &mut [Move]) {
    order_moves_with_tt_and_killers(board, moves, None, [None, None]);
}

/// Order captures for quiescence search (MVV-LVA only)
pub fn order_captures(board: &Board, moves: &mut [Move]) {
    let mut scored: Vec<(Move, i32)> = moves.iter()
        .map(|&m| (m, mvv_lva_score(board, m)))
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1));

    for (i, (m, _)) in scored.into_iter().enumerate() {
        moves[i] = m;
    }
}
