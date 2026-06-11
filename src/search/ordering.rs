//! Move ordering heuristics.
//!
//! Good move ordering is critical for alpha-beta pruning efficiency.
//! Uses lazy selection sort to avoid full sort overhead.

use super::history::HistoryTable;
use super::see;
use crate::types::{piece_value, Board, Color, Move};

/// Move score constants
const TT_MOVE_BONUS: i32 = 1_000_000;
const PROMOTION_BONUS: i32 = 100_000;
const GOOD_CAPTURE_BONUS: i32 = 60_000;
const KILLER_0_BONUS: i32 = 40_000;
const KILLER_1_BONUS: i32 = 35_000;
const COUNTER_MOVE_BONUS: i32 = 30_000;
const BAD_CAPTURE_PENALTY: i32 = -10_000;

/// Score a move for ordering (higher = search first)
#[inline]
pub fn score_move(
    board: &Board,
    m: Move,
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
    counter_move: Option<Move>,
    history: &HistoryTable,
    color: Color,
) -> i32 {
    // TT move is always searched first
    if tt_move == Some(m) {
        return TT_MOVE_BONUS;
    }

    let mut score = 0;

    // Promotions are very important
    if let Some(promo) = m.flag().promotion_piece() {
        score += piece_value(promo) + PROMOTION_BONUS;
    }

    // Captures: skip SEE for obviously good captures (victim >= attacker)
    if m.is_capture() {
        // MVV-LVA logic inlined to reuse victim for SEE
        let victim = board.piece_at(m.to()).map(|(p, _)| p);
        let attacker = board.piece_at(m.from()).map(|(p, _)| p);

        let mvv_lva = match (victim, attacker) {
            (Some(v), Some(a)) => piece_value(v) * 10 - piece_value(a),
            _ => 0,
        };

        if mvv_lva >= 0 {
            // Winning or equal capture (e.g., PxQ, NxN) - skip expensive SEE
            score += GOOD_CAPTURE_BONUS + mvv_lva;
        } else {
            // Potentially losing capture - use SEE to verify
            // Pass the victim we already found to avoid re-lookup
            let see_value = see::see_captured(board, m, victim);
            if see_value >= 0 {
                score += GOOD_CAPTURE_BONUS + mvv_lva;
            } else {
                score += BAD_CAPTURE_PENALTY + mvv_lva;
            }
        }
    } else {
        // Quiet move - check killers and counter-move
        if killers[0] == Some(m) {
            score += KILLER_0_BONUS;
        } else if killers[1] == Some(m) {
            score += KILLER_1_BONUS;
        } else if counter_move == Some(m) {
            score += COUNTER_MOVE_BONUS;
        } else {
            // Use history score for other quiet moves
            score += history.get(color, m);
        }
    }

    score
}

use crate::types::MoveList;

pub struct MovePicker<'a> {
    board: &'a Board,
    moves: MoveList,
    scores: [i32; 256],
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
    counter_move: Option<Move>,
    color: Color,

    phase: i32,
    yielded_killers: usize,
}

impl<'a> MovePicker<'a> {
    pub fn new(
        board: &'a Board,
        moves: MoveList,
        tt_move: Option<Move>,
        killers: [Option<Move>; 2],
        counter_move: Option<Move>,
        color: Color,
    ) -> Self {
        // Verify TT move is legal by checking if it's in the generated move list
        let valid_tt = if let Some(tt) = tt_move {
            moves.iter().any(|m| m == tt)
        } else {
            false
        };
        let validated_tt = if valid_tt { tt_move } else { None };

        Self {
            board,
            moves,
            scores: [0; 256],
            tt_move: validated_tt,
            killers,
            counter_move,
            color,
            phase: 0,
            yielded_killers: 0,
        }
    }

    pub fn next(&mut self, history: &HistoryTable) -> Option<Move> {
        loop {
            match self.phase {
                0 => {
                    // Phase 1: TT Move
                    self.phase = 1;
                    if let Some(tt) = self.tt_move {
                        if let Some(idx) = self.moves.iter().position(|m| m == tt) {
                            self.scores[idx] = i32::MIN; // Mark as yielded
                        }
                        return Some(tt);
                    }
                }
                1 => {
                    // Score captures
                    for i in 0..self.moves.len() {
                        let m = self.moves.as_slice()[i];
                        if self.scores[i] != i32::MIN && (m.is_capture() || m.is_promotion()) {
                            self.scores[i] = score_move(
                                self.board,
                                m,
                                None,
                                [None, None],
                                None,
                                history,
                                self.color,
                            );
                        }
                    }
                    self.phase = 2;
                }
                2 => {
                    // Phase 2: Yield captures iteratively
                    let mut best_score = -i32::MAX;
                    let mut best_idx = None;

                    for i in 0..self.moves.len() {
                        let m = self.moves.as_slice()[i];
                        if self.scores[i] != i32::MIN && (m.is_capture() || m.is_promotion()) {
                            if self.scores[i] > best_score {
                                best_score = self.scores[i];
                                best_idx = Some(i);
                            }
                        }
                    }

                    if let Some(idx) = best_idx {
                        self.scores[idx] = i32::MIN; // Mark as yielded
                        return Some(self.moves.as_slice()[idx]);
                    } else {
                        self.phase = 3;
                    }
                }
                3 => {
                    // Phase 3: Yield killers
                    while self.yielded_killers < 2 {
                        let k = self.killers[self.yielded_killers];
                        self.yielded_killers += 1;

                        if let Some(killer) = k {
                            if Some(killer) != self.tt_move {
                                if let Some(idx) = self.moves.iter().position(|m| {
                                    m == killer && !m.is_capture() && !m.is_promotion()
                                }) {
                                    if self.scores[idx] != i32::MIN {
                                        self.scores[idx] = i32::MIN; // Mark as yielded
                                        return Some(killer);
                                    }
                                }
                            }
                        }
                    }
                    self.phase = 4;
                }
                4 => {
                    // Score quiets
                    for i in 0..self.moves.len() {
                        let m = self.moves.as_slice()[i];
                        if self.scores[i] != i32::MIN && !m.is_capture() && !m.is_promotion() {
                            self.scores[i] = score_move(
                                self.board,
                                m,
                                None,
                                [None, None],
                                self.counter_move,
                                history,
                                self.color,
                            );
                        }
                    }
                    self.phase = 5;
                }
                5 => {
                    // Phase 4: Yield quiets iteratively
                    let mut best_score = -i32::MAX;
                    let mut best_idx = None;

                    for i in 0..self.moves.len() {
                        let m = self.moves.as_slice()[i];
                        if self.scores[i] != i32::MIN && !m.is_capture() && !m.is_promotion() {
                            if self.scores[i] > best_score {
                                best_score = self.scores[i];
                                best_idx = Some(i);
                            }
                        }
                    }

                    if let Some(idx) = best_idx {
                        self.scores[idx] = i32::MIN; // Mark as yielded
                        return Some(self.moves.as_slice()[idx]);
                    } else {
                        return None; // Done
                    }
                }
                _ => return None,
            }
        }
    }
}
