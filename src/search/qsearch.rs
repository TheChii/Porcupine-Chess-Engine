//! Quiescence search - search captures only to avoid horizon effect.
//!
//! When the main search reaches depth 0, we continue searching captures
//! to ensure we don't stop in the middle of a tactical sequence.

use super::{Searcher, ordering};
use super::negamax::SearchResult;
use crate::types::{Board, Move, Score, Ply, MoveGen};
use crate::eval::SearchEvaluator;
use arrayvec::ArrayVec;
use std::time::Instant;

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
    searcher.inc_qnodes();
    searcher.update_seldepth(ply);

    // Stand-pat evaluation using incremental evaluator
    searcher.inc_eval_calls();
    let t_eval = Instant::now();
    let stand_pat = evaluator.evaluate(board);
    searcher.add_eval_time(t_eval.elapsed().as_nanos() as u64);

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

    // Generate capture moves only - use ArrayVec
    let t_gen = Instant::now();
    let mut moves: ArrayVec<Move, 64> = MoveGen::new_legal(board)
        .filter(|m| board.piece_on(m.get_dest()).is_some())
        .take(64)
        .collect();
    searcher.add_gen_time(t_gen.elapsed().as_nanos() as u64);

    if moves.is_empty() {
        return SearchResult {
            best_move: None,
            score: alpha,
            pv: Vec::new(),
            stats: searcher.stats().clone(),
        };
    }

    let t_order = Instant::now();
    ordering::order_captures(board, &mut moves);
    searcher.add_order_time(t_order.elapsed().as_nanos() as u64);

    let mut best_score = stand_pat;
    let mut pv = Vec::new();

    for i in 0..moves.len() {
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
