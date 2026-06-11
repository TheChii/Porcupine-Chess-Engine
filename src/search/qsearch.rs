//! Quiescence search - search captures only to avoid horizon effect.
//!
//! When the main search reaches depth 0, we continue searching captures
//! to ensure we don't stop in the middle of a tactical sequence.
//!
//! Implements delta pruning to skip hopeless captures.
//!
//! Uses compile-time node type specialization via the `NodeType` trait.

use super::negamax::SearchResult;
use super::node_types::NodeType;
use super::see::is_good_capture_with_victim;
use super::{ordering, Searcher};
use crate::eval::SearchEvaluator;
use crate::types::{Board, Piece, Ply, Score};

/// Piece values for delta pruning (centipawns)
const PIECE_VALUES: [i32; 6] = [
    100, // Pawn
    320, // Knight
    330, // Bishop
    500, // Rook
    900, // Queen
    0,   // King (never captured)
];

/// Delta margin: if stand_pat + best possible gain < alpha, prune
/// Using Queen value as the maximum possible gain from a single capture
const DELTA_MARGIN: i32 = 600;

/// Safety margin for individual move delta pruning
const DELTA_SAFETY: i32 = 100;

/// Maximum depth for quiescence search (beyond main search)
/// After this depth, only continue if in check
const MAX_QSEARCH_DEPTH: i32 = 8;

/// Get the value of a piece for delta pruning
#[inline]
fn piece_value(piece: Piece) -> i32 {
    PIECE_VALUES[piece.index()]
}

/// Quiescence search - search captures only to avoid horizon effect.
///
/// Uses compile-time node type specialization via the `NodeType` trait.
/// `qply` tracks depth within qsearch (starts at 0).
pub fn quiescence<NT: NodeType>(
    searcher: &mut Searcher,
    evaluator: &mut SearchEvaluator,
    board: &Board,
    ply: Ply,
    qply: i32,
    mut alpha: Score,
    beta: Score,
) -> SearchResult {
    searcher.inc_nodes();
    searcher.inc_qnodes();
    searcher.update_seldepth(ply);

    let p = ply.raw() as usize;
    searcher.pv_length[p] = 0;

    let in_check = board.in_check();

    // Stand-pat evaluation using incremental evaluator
    let stand_pat = if in_check {
        Score::neg_infinity()
    } else {
        #[cfg(debug_assertions)]
        searcher.inc_eval_calls();
        #[cfg(debug_assertions)]
        let t_eval = std::time::Instant::now();
        let eval = evaluator.evaluate(ply.raw() as usize, board);
        #[cfg(debug_assertions)]
        searcher.add_eval_time(t_eval.elapsed().as_nanos() as u64);
        eval
    };

    // Beta cutoff: position is already too good (only if not in check)
    if !in_check && stand_pat >= beta {
        return SearchResult {
            best_move: None,
            score: beta,
        };
    }

    // === Delta Pruning (Big Delta) ===
    // If even capturing a queen wouldn't bring us close to alpha, give up

    // === Qsearch Depth Limit ===
    // Beyond MAX_QSEARCH_DEPTH, only continue if in check
    if qply >= MAX_QSEARCH_DEPTH && !in_check {
        return SearchResult {
            best_move: None,
            score: stand_pat,
        };
    }

    if !in_check && stand_pat.raw() + DELTA_MARGIN < alpha.raw() {
        return SearchResult {
            best_move: None,
            score: alpha,
        };
    }

    let mut best_score = stand_pat;

    if in_check {
        best_score = Score::mated_in(ply.raw()); // Base mate score if no evasions
    } else if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Generate moves: all evasions if in check, otherwise only captures
    #[cfg(debug_assertions)]
    let t_gen = std::time::Instant::now();
    let moves = if in_check {
        board.generate_moves()
    } else {
        board.generate_captures()
    };
    #[cfg(debug_assertions)]
    searcher.add_gen_time(t_gen.elapsed().as_nanos() as u64);

    if moves.is_empty() {
        if in_check {
            return SearchResult {
                best_move: None,
                score: Score::mated_in(ply.raw()),
            };
        } else {
            return SearchResult {
                best_move: None,
                score: alpha,
            };
        }
    }

    let mut move_picker = ordering::CapturePicker::new(board, moves);

    while let Some(m) = move_picker.next() {
        if searcher.should_stop() {
            break;
        }

        // Get captured piece value for delta pruning
        let captured = board.piece_at(m.to()).map(|(p, _)| p);
        let captured_value = captured.map(piece_value).unwrap_or(0);

        // === Delta Pruning (Per-Move) ===
        // If this capture + safety margin can't raise alpha, skip it
        // Skip this check for promotions (they gain material)
        if !in_check
            && !m.is_promotion()
            && stand_pat.raw() + captured_value + DELTA_SAFETY < alpha.raw()
        {
            continue;
        }

        // === SEE Pruning ===
        // Skip captures that lose material according to SEE
        if !in_check && !is_good_capture_with_victim(board, m, captured) {
            continue;
        }

        let new_board = board.make_move_new(m);

        // Update evaluator incrementally for next depth
        if !evaluator.update_move(ply.raw() as usize, board, m) {
            evaluator.refresh(ply.next().raw() as usize, &new_board);
        }

        let result = quiescence::<NT::Next>(
            searcher,
            evaluator,
            &new_board,
            ply.next(),
            qply + 1,
            -beta,
            -alpha,
        );
        let score = -result.score;

        if score > best_score {
            best_score = score;

            // Update Triangular PV Table
            searcher.pv_table[p][0] = m;
            let next_p = p + 1;
            let len = searcher.pv_length[next_p];
            for i in 0..len {
                searcher.pv_table[p][i + 1] = searcher.pv_table[next_p][i];
            }
            searcher.pv_length[p] = len + 1;

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
    }
}
