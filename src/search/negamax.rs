//! Negamax alpha-beta search implementation.
//!
//! This is the core search algorithm with:
//! - Transposition table probing and storing
//! - Alpha-beta pruning
//! - Quiescence search for captures
//! - Compile-time node type specialization (no runtime PV checks)
//!
//! Uses Rust generics for compile-time node type specialization.
//! See `node_types` module for `NodeType` trait and concrete types.

use super::node_types::{NodeType, OffPV};
use super::tt::BoundType;
use super::{ordering, qsearch, see, Searcher};
use crate::eval::SearchEvaluator;
use crate::types::{Board, Depth, Move, Piece, Ply, Score, SCORE_MATE};

use std::sync::OnceLock;

static LMR_TABLE: OnceLock<[[i32; 64]; 64]> = OnceLock::new();

#[inline]
fn get_lmr(depth: i32, move_idx: usize) -> i32 {
    let table = LMR_TABLE.get_or_init(|| {
        let mut t = [[0; 64]; 64];
        for d in 0..64 {
            for m in 0..64 {
                let d_f = (d.max(1) as f32).ln();
                let m_f = ((m + 1) as f32).ln();
                t[d][m] = (0.6 + d_f * m_f / 1.6) as i32;
            }
        }
        t
    });

    let d = depth.min(63).max(0) as usize;
    let m = move_idx.min(63);
    table[d][m]
}

/// Result from a search
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: Score,
}

/// Main negamax search function with TT integration and null move pruning.
///
/// Uses compile-time node type specialization via the `NodeType` trait.
/// - `NT::PV`: true if this is a principal variation node
/// - `NT::ROOT`: true if this is the root node
/// - `NT::Next`: the node type for child PV searches
pub fn search<NT: NodeType>(
    searcher: &mut Searcher,
    evaluator: &mut SearchEvaluator,
    board: &Board,
    depth: Depth,
    ply: Ply,
    mut alpha: Score,
    mut beta: Score,
    prev_move: Option<Move>,
) -> SearchResult {
    searcher.inc_nodes();
    searcher.update_seldepth(ply);

    let p = ply.raw() as usize;
    searcher.pv_length[p] = 0;

    let hash = board.hash();

    // === Repetition Detection with Contempt ===
    // Check for draw by repetition (position seen before in game history)
    // Use contempt: avoid draws when winning, seek draws when losing
    // Skip at root node (ply == 0)
    if !NT::ROOT && searcher.is_repetition(hash) {
        // Contempt factor: small penalty/bonus for draws based on expected score
        // If alpha > 0 (we expect to be winning), penalize draws to avoid them
        // If beta < 0 (we expect to be losing), reward draws to seek them
        const CONTEMPT: i32 = 10; // Small contempt factor (centipawns)

        let draw_score = if alpha.raw() > CONTEMPT {
            // We're winning - penalize draws to avoid repetition
            Score::cp(-CONTEMPT)
        } else if beta.raw() < -CONTEMPT {
            // We're losing - reward draws to seek repetition
            Score::cp(CONTEMPT)
        } else {
            // Close to equal - treat as pure draw
            Score::draw()
        };

        return SearchResult {
            best_move: None,
            score: draw_score,
        };
    }

    // Mate distance pruning
    let mate_score = SCORE_MATE - ply.raw();
    let mated_score = -SCORE_MATE + ply.raw();

    if alpha.raw() < mated_score {
        alpha = Score(mated_score as i16);
        if alpha >= beta {
            return SearchResult {
                best_move: None,
                score: alpha,
                };
        }
    }

    if beta.raw() > mate_score {
        beta = Score(mate_score as i16);
        if alpha >= beta {
            return SearchResult {
                best_move: None,
                score: beta,
                };
        }
    }

    let orig_alpha = alpha;
    let mut tt_move: Option<Move> = None;

    // === TT Probe ===
    if let Some(entry) = searcher.shared.tt.probe(hash) {
        tt_move = entry.best_move();

        // Only use TT score if depth is sufficient
        if entry.depth() >= depth {
            let tt_score = entry.score().from_tt(ply.raw());

            match entry.bound() {
                BoundType::Exact => {
                    if let Some(m) = tt_move {
                        searcher.pv_table[p][0] = m;
                        searcher.pv_length[p] = 1;
                    }
                    return SearchResult {
                        best_move: tt_move,
                        score: tt_score,
                                };
                }
                BoundType::LowerBound => {
                    if tt_score >= beta {
                        if let Some(m) = tt_move {
                            searcher.pv_table[p][0] = m;
                            searcher.pv_length[p] = 1;
                        }
                        return SearchResult {
                            best_move: tt_move,
                            score: tt_score,
                                        };
                    }
                    // Tighten alpha only for non-PV nodes
                    if !NT::PV && tt_score > alpha {
                        alpha = tt_score;
                    }
                }
                BoundType::UpperBound => {
                    if tt_score <= alpha {
                        if let Some(m) = tt_move {
                            searcher.pv_table[p][0] = m;
                            searcher.pv_length[p] = 1;
                        }
                        return SearchResult {
                            best_move: tt_move,
                            score: tt_score,
                                        };
                    }
                    // Tighten beta only for non-PV nodes
                    if !NT::PV && tt_score < beta {
                        beta = tt_score;
                    }
                }
                BoundType::None => {}
            }
        }
    }

    // Check for stop condition
    if searcher.should_stop() {
        return SearchResult {
            best_move: None,
            score: Score::draw(),
        };
    }

    let in_check = board.in_check();

    // === Internal Iterative Deepening (IID) ===
    // If we have no TT move at a high depth, do a shallower search to find a good move
    // This dramatically improves move ordering and alpha-beta cutoffs for the main search
    if tt_move.is_none() && depth.raw() >= 5 && (NT::PV || depth.raw() >= 6) {
        let iid_depth = Depth::new(depth.raw() - 2);
        
        // Do a shallower search to populate TT
        let _ = search::<NT>(
            searcher,
            evaluator,
            board,
            iid_depth,
            ply,
            alpha,
            beta,
            prev_move,
        );
        
        // Probe TT again to get the move
        if let Some(entry) = searcher.shared.tt.probe(hash) {
            tt_move = entry.best_move();
        }
    }
    
    // Fallback Reductions (IIR) for when IID isn't run
    let mut iir_reduction = 0;
    if tt_move.is_none() && depth.raw() >= 4 {
        iir_reduction = 1;
    }

    let adjusted_depth = Depth::new((depth.raw() - iir_reduction).max(0));

    // === Reverse Futility Pruning (RFP) ===
    // If we are way ahead, we can prune without searching
    // Distinct from standard Futility Pruning which prunes *moves*
    let mut static_eval = None;
    let pawn_hash = board.pawn_hash();
    let color = board.turn();

    if !in_check && adjusted_depth.raw() <= 8 {
        #[cfg(debug_assertions)]
        searcher.inc_eval_calls();
        #[cfg(debug_assertions)]
        let t_eval = std::time::Instant::now();
        let raw_eval = evaluator.evaluate(ply.raw() as usize, board);
        #[cfg(debug_assertions)]
        searcher.add_eval_time(t_eval.elapsed().as_nanos() as u64);

        let eval = raw_eval;
        static_eval = Some(eval);

        // RFP Margin: 75 * depth
        let margin = Score::cp(75 * adjusted_depth.raw());

        if eval - margin >= beta {
            return SearchResult {
                best_move: None,
                score: beta, // Fail high directly
                };
        }
    }

    // === ProbCut ===
    // Only on non-PV nodes (zero-window)
    const PROBCUT_MARGIN: i32 = 100;
    if !NT::PV && adjusted_depth.raw() >= 5 && !in_check && beta.raw().abs() < (SCORE_MATE - 1000) {
        let probe_beta = beta + Score::cp(PROBCUT_MARGIN);
        let probe_depth = Depth::new(adjusted_depth.raw() - 4);

        let result = search::<OffPV>(
            searcher,
            evaluator,
            board,
            probe_depth,
            ply,
            probe_beta - Score::cp(1),
            probe_beta,
            None,
        );

        if result.score >= probe_beta {
            return SearchResult {
                best_move: result.best_move,
                score: beta,
                };
        }
    }

    // === Null Move Pruning ===
    // Skip if: in check, depth too low, PV node, or only king+pawns
    // Note: we don't do NMP on PV nodes or at root
    if !NT::PV && !in_check && adjusted_depth.raw() >= 3 {
        // Don't do null move in pure pawn endgames (zugzwang risk)
        let dominated_by_pawns = (board.piece_bb(Piece::Knight)
            | board.piece_bb(Piece::Bishop)
            | board.piece_bb(Piece::Rook)
            | board.piece_bb(Piece::Queen))
        .is_empty();

        if !dominated_by_pawns {
            // Reduction: Base 3 + scaled by depth
            let r = 3 + adjusted_depth.raw() / 6;

            // Create a null move board (pass the turn)
            let null_board = board.make_null_move();

            // Use current evaluator, just refresh the next ply for null board
            evaluator.refresh(ply.next().raw() as usize, &null_board);

            let null_result = search::<OffPV>(
                searcher,
                evaluator,
                &null_board,
                Depth::new((adjusted_depth.raw() - 1 - r).max(0)),
                ply.next(),
                -beta,
                -beta + Score::cp(1),
                None, // No prev move for null move
            );

            let null_score = -null_result.score;

            if null_score >= beta {
                // Null move cutoff
                return SearchResult {
                    best_move: None,
                    score: beta, // Fail high
                        };
            }
        }
    }

    // Check for checkmate or stalemate early?
    // We can't anymore because we defer move generation! Wait, no, we generate moves upfront!
    #[cfg(debug_assertions)]
    let t_gen = std::time::Instant::now();
    let moves = board.generate_moves();
    #[cfg(debug_assertions)]
    searcher.add_gen_time(t_gen.elapsed().as_nanos() as u64);

    // Check for checkmate or stalemate
    if moves.is_empty() {
        let score = if board.in_check() {
            Score::mated_in(ply.raw())
        } else {
            Score::draw()
        };
        return SearchResult {
            best_move: None,
            score,
        };
    }

    // Quiescence search at depth 0
    if adjusted_depth.is_qs() {
        return qsearch::quiescence::<NT>(searcher, evaluator, board, ply, 0, alpha, beta);
    }

    // Get killers for this ply
    let killers = searcher.killers.get(ply);
    let color = board.turn();

    // Get counter-move for opponent's previous move
    let counter_move = prev_move.and_then(|pm| searcher.countermoves.get(pm));

    // Create MovePicker to lazily score and yield moves
    let mut move_picker =
        ordering::MovePicker::new(board, moves, tt_move, killers, counter_move, color);

    // Static eval is already computed for RFP if depth <= 7
    // If not (e.g. was in check check or deeper), compute it now if needed for Razoring/Futility
    if static_eval.is_none() && depth.raw() <= 3 && !in_check {
        #[cfg(debug_assertions)]
        searcher.inc_eval_calls();
        #[cfg(debug_assertions)]
        let t_eval = std::time::Instant::now();
        let val = evaluator.evaluate(ply.raw() as usize, board);
        #[cfg(debug_assertions)]
        searcher.add_eval_time(t_eval.elapsed().as_nanos() as u64);
        static_eval = Some(val);
    }

    // Razoring - only on non-PV nodes
    if !NT::PV && depth.raw() <= 3 && !in_check {
        if let Some(eval) = static_eval {
            let threshold = alpha - Score::cp(200 + depth.raw() * 60);
            if eval < threshold {
                let result =
                    qsearch::quiescence::<OffPV>(searcher, evaluator, board, ply, 0, alpha, beta);
                if result.score < alpha {
                    return result;
                }
            }
        }
    }

    let mut best_move = None;
    let mut best_score = Score::neg_infinity();
    // Use fixed-size array for searched quiets to avoid allocations
    let mut searched_quiets: [Move; 64] = [Move::NULL; 64];
    let mut quiets_count = 0usize;

    let mut move_idx = 0;
    while let Some(m) = move_picker.next(&searcher.history) {
        if NT::ROOT {
            searcher.report_currmove(m, move_idx + 1);
        }

        let new_board = board.make_move_new(m);

        // Prefetch TT entry for next position
        searcher.shared.tt.prefetch(new_board.hash());

        // Determine if this is a quiet move (for LMR)
        let is_capture = m.is_capture();
        let is_promotion = m.is_promotion();
        let is_killer = killers[0] == Some(m) || killers[1] == Some(m);
        let is_quiet = !is_capture && !is_promotion;
        let gives_check = new_board.in_check();
        let is_good_capture = is_capture && see::see_ge(board, m, 0);

        // === Late Move Pruning (LMP) ===
        // If we have searched enough quiet moves at low depth, stop searching the rest.
        // This relies on move ordering to put good moves early.
        if is_quiet && adjusted_depth.raw() <= 7 && !in_check {
            // Formula: LMS = 3 + depth^2 (e.g., d1=4, d2=7, d3=12...)
            let lmp_count = (3 + adjusted_depth.raw() * adjusted_depth.raw()) as usize;
            if quiets_count > lmp_count {
                continue;
            }
        }

        // LMR: Late Move Reductions
        // Reduce depth for late quiet moves that aren't special
        let mut reduced = false;

        // Check extension: extend +1 when in check to avoid horizon effect
        let extension = if in_check { 1 } else { 0 };

        let search_depth = if move_idx >= 2
            && adjusted_depth.raw() >= 3
            && !in_check
            && !gives_check
            && !is_killer
            && !is_good_capture
        {
            // Logarithmic reduction formula (pre-computed)
            let mut reduction = get_lmr(adjusted_depth.raw(), move_idx);

            // Reduce more for quiet moves
            if is_quiet {
                reduction += 1;
            }

            // History-based LMR adjustment
            let history_score = searcher.history.get(color, m);
            if history_score < -15000 {
                reduction += 1;
            } else if history_score > 15000 {
                reduction -= 1;
            }

            let reduction = reduction.min(adjusted_depth.raw() - 1).max(1);
            reduced = true;
            Depth::new((adjusted_depth.raw() - 1 - reduction + extension).max(1))
        } else {
            Depth::new((adjusted_depth.raw() - 1 + extension).max(0))
        };

        // === History Pruning ===
        // Prune quiet moves that have historically failed significantly
        if adjusted_depth.raw() < 6
            && is_quiet
            && !in_check
            && !gives_check
            && !is_killer
            && move_idx > 0
        {
            // More aggressive pruning threshold: -2000 * depth
            let threshold = -2000 * adjusted_depth.raw();
            if searcher.history.get(color, m) < threshold {
                continue;
            }
        }

        // === SEE Pruning for Quiet Moves ===
        // Prune quiet moves that are obvious blunders (e.g. putting a piece en prise)
        if adjusted_depth.raw() <= 4 && is_quiet && !in_check && !gives_check && move_idx > 0 {
            // If move loses material (at least 50cp), prune it
            // This uses SEE to see if the move is "safe"
            if !see::see_ge(board, m, -50) {
                continue;
            }
        }

        // === Futility Pruning ===
        // At shallow depths, skip quiet moves if eval + margin is below alpha
        if let Some(se) = static_eval {
            if is_quiet && !gives_check && move_idx > 0 {
                // More aggressive margin: 120 * depth (was 90)
                let margin = 120 * adjusted_depth.raw();
                if se.raw() + margin < alpha.raw() {
                    // Track for history
                    if quiets_count < 64 {
                        searched_quiets[quiets_count] = m;
                        quiets_count += 1;
                    }
                    continue; // Prune this move
                }
            }
        }

        // === SEE Pruning for Captures ===
        // Prune captures that are obviously losing at shallow depths
        if adjusted_depth.raw() <= 5 && is_capture && move_idx > 0 {
            // Threshold becomes more lenient with depth: -100 * depth
            let threshold = -100 * adjusted_depth.raw();
            if !see::see_ge(board, m, threshold) {
                continue;
            }
        }

        // === Principal Variation Search (PVS) ===
        let mut result;
        let mut score;

        if move_idx == 0 {
            // Incremental update for next depth
            if !evaluator.update_move(ply.raw() as usize, board, m) {
                evaluator.refresh(ply.next().raw() as usize, &new_board);
            }

            // First move: search with full window (PV search)
            result = search::<NT::Next>(
                searcher,
                evaluator,
                &new_board,
                search_depth,
                ply.next(),
                -beta,
                -alpha,
                Some(m), // Pass current move as prev_move
            );
            score = -result.score;
        } else {
            // Incremental update
            if !evaluator.update_move(ply.raw() as usize, board, m) {
                evaluator.refresh(ply.next().raw() as usize, &new_board);
            }

            // Later moves: null window search first (OffPV)
            result = search::<OffPV>(
                searcher,
                evaluator,
                &new_board,
                search_depth,
                ply.next(),
                -alpha - Score::cp(1),
                -alpha,
                Some(m),
            );
            score = -result.score;

            // Re-search with full window if fails high (only on PV nodes)
            if NT::PV && score > alpha && score < beta && !searcher.should_stop() {
                // Re-use same evaluator since board/move didn't change
                result = search::<NT::Next>(
                    searcher,
                    evaluator,
                    &new_board,
                    search_depth,
                    ply.next(),
                    -beta,
                    -alpha,
                    Some(m),
                );
                score = -result.score;
            }
        }

        // Re-search at full depth if LMR reduced search beats alpha
        if reduced && score > alpha && !searcher.should_stop() {
            if !evaluator.update_move(ply.raw() as usize, board, m) {
                evaluator.refresh(ply.next().raw() as usize, &new_board);
            }

            // First: null-window re-search at full depth
            result = search::<OffPV>(
                searcher,
                evaluator,
                &new_board,
                Depth::new((depth.raw() - 1 + extension).max(0)),
                ply.next(),
                -alpha - Score::cp(1),
                -alpha,
                Some(m),
            );
            score = -result.score;

            // If null-window also fails high on PV nodes, do full-window re-search
            if NT::PV && score > alpha && score < beta && !searcher.should_stop() {
                result = search::<NT::Next>(
                    searcher,
                    evaluator,
                    &new_board,
                    Depth::new((depth.raw() - 1 + extension).max(0)),
                    ply.next(),
                    -beta,
                    -alpha,
                    Some(m),
                );
                score = -result.score;
            }
        }

        if searcher.should_stop() {
            break;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m);

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
                    // Beta cutoff - update killer, history, and counter-move for quiet moves
                    if is_quiet {
                        searcher.killers.store(ply, m);
                        // Update history: bonus for cutoff move, penalty for searched quiets
                        searcher.history.update_on_cutoff(
                            color,
                            m,
                            depth.raw(),
                            &searched_quiets[..quiets_count],
                        );
                        // Update counter-move
                        if let Some(pm) = prev_move {
                            searcher.countermoves.store(pm, m);
                        }
                    }
                    break;
                }
            }
        }

        // Track searched quiet moves for history penalty
        if is_quiet && quiets_count < 64 {
            searched_quiets[quiets_count] = m;
            quiets_count += 1;
        }

        move_idx += 1;
    }

    // === TT Store ===
    if !searcher.should_stop() {
        let bound = if best_score >= beta {
            BoundType::LowerBound
        } else if best_score > orig_alpha {
            BoundType::Exact
        } else {
            BoundType::UpperBound
        };

        searcher
            .shared
            .tt
            .store(hash, best_move, best_score.to_tt(ply.raw()), depth, bound);
    }

    SearchResult {
        best_move,
        score: best_score,
    }
}
