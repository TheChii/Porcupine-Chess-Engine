//! Quiescence search - search captures only to avoid horizon effect.
//!
//! When the main search reaches depth 0, we continue searching captures
//! to ensure we don't stop in the middle of a tactical sequence.

use super::{Searcher, ordering};
use super::negamax::SearchResult;
use crate::types::{Board, Move, Score, Ply, MoveGen};
use crate::eval::SearchEvaluator;

/// Quiescence search - search captures only to avoid horizon effect
pub fn quiescence(
    searcher: &mut Searcher,
    evaluator: &mut SearchEvaluator,
    board: &Board,
    ply: Ply,
    mut alpha: Score,
    beta: Score,
) -> SearchResult {
    searcher.inc_nodes();
    searcher.update_seldepth(ply);

    // Stand-pat evaluation using incremental evaluator
    let stand_pat = evaluator.evaluate(board);

    if stand_pat >= beta {
        return SearchResult {
            best_move: None,
            score: beta,
            pv: Vec::new(),
            stats: searcher.stats().clone(),
        };
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Generate capture moves only - use fixed array
    let mut moves: [Move; 64] = [Move::default(); 64];
    let mut move_count = 0;
    
    for m in MoveGen::new_legal(board) {
        if board.piece_on(m.get_dest()).is_some() && move_count < 64 {
            moves[move_count] = m;
            move_count += 1;
        }
    }

    if move_count == 0 {
        return SearchResult {
            best_move: None,
            score: alpha,
            pv: Vec::new(),
            stats: searcher.stats().clone(),
        };
    }

    ordering::order_captures(board, &mut moves[..move_count]);

    let mut best_score = stand_pat;
    let mut pv = Vec::new();

    for i in 0..move_count {
        let m = moves[i];
        if searcher.should_stop() {
            break;
        }

        let new_board = board.make_move_new(m);
        
        // Clone evaluator for next depth and update incrementally
        let mut child_evaluator = evaluator.clone();
        child_evaluator.update_move(board, m); // board is position BEFORE move

        let result = quiescence(searcher, &mut child_evaluator, &new_board, ply.next(), -beta, -alpha);
        let score = -result.score;

        if score > best_score {
            best_score = score;

            pv.clear();
            pv.push(m);
            pv.extend(result.pv);

            if score > alpha {
                alpha = score;
                if score >= beta {
                    break;
                }
            }
        }
    }

    SearchResult {
        best_move: None,
        score: best_score,
        pv,
        stats: searcher.stats().clone(),
    }
}
