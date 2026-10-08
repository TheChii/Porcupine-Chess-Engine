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

use std::mem::MaybeUninit;
use crate::types::MoveList;

pub struct MovePicker<'a> {
    board: &'a Board,
    moves: MoveList,
    scores: [MaybeUninit<i32>; 256],
    num_tactical: usize,
    cur_tactical: usize,
    cur_quiet: usize,
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
        mut moves: MoveList,
        tt_move: Option<Move>,
        killers: [Option<Move>; 2],
        counter_move: Option<Move>,
        color: Color,
    ) -> Self {
        let valid_tt = if let Some(tt) = tt_move {
            moves.contains(tt)
        } else {
            false
        };
        let validated_tt = if valid_tt { tt_move } else { None };

        // Partition moves into tactical (captures + promotions) and quiets
        let num_moves = moves.len();
        let slice = moves.as_slice_mut();
        let mut num_tactical = 0;
        for i in 0..num_moves {
            if slice[i].is_capture() || slice[i].is_promotion() {
                slice.swap(i, num_tactical);
                num_tactical += 1;
            }
        }

        Self {
            board,
            moves,
            scores: [MaybeUninit::uninit(); 256],
            num_tactical,
            cur_tactical: 0,
            cur_quiet: num_tactical,
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
                    // Phase 0: TT Move
                    self.phase = 1;
                    if let Some(tt) = self.tt_move {
                        return Some(tt);
                    }
                }
                1 => {
                    // Phase 1: Score tactical moves
                    for i in 0..self.num_tactical {
                        let m = self.moves.as_slice()[i];
                        if Some(m) == self.tt_move {
                            self.scores[i].write(i32::MIN);
                        } else {
                            let s = score_move(
                                self.board,
                                m,
                                None,
                                [None, None],
                                None,
                                history,
                                self.color,
                            );
                            self.scores[i].write(s);
                        }
                    }
                    self.phase = 2;
                }
                2 => {
                    // Phase 2: Yield tactical moves via swap-selection
                    while self.cur_tactical < self.num_tactical {
                        let mut best_score = unsafe { self.scores[self.cur_tactical].assume_init() };
                        let mut best_idx = self.cur_tactical;

                        for i in (self.cur_tactical + 1)..self.num_tactical {
                            let sc = unsafe { self.scores[i].assume_init() };
                            if sc > best_score {
                                best_score = sc;
                                best_idx = i;
                            }
                        }

                        if best_idx != self.cur_tactical {
                            self.moves.as_slice_mut().swap(self.cur_tactical, best_idx);
                            self.scores.swap(self.cur_tactical, best_idx);
                        }

                        let m = self.moves.as_slice()[self.cur_tactical];
                        self.cur_tactical += 1;

                        if best_score != i32::MIN {
                            return Some(m);
                        }
                    }
                    self.phase = 3;
                }
                3 => {
                    // Phase 3: Yield killers
                    while self.yielded_killers < 2 {
                        let k = self.killers[self.yielded_killers];
                        self.yielded_killers += 1;

                        if let Some(killer) = k {
                            if Some(killer) != self.tt_move {
                                if let Some(pos) = (self.cur_quiet..self.moves.len())
                                    .find(|&i| self.moves.as_slice()[i] == killer)
                                {
                                    self.moves.as_slice_mut().swap(self.cur_quiet, pos);
                                    let m = self.moves.as_slice()[self.cur_quiet];
                                    self.cur_quiet += 1;
                                    return Some(m);
                                }
                            }
                        }
                    }
                    self.phase = 4;
                }
                4 => {
                    // Phase 4: Score remaining quiet moves
                    let total_moves = self.moves.len();
                    for i in self.cur_quiet..total_moves {
                        let m = self.moves.as_slice()[i];
                        if Some(m) == self.tt_move {
                            self.scores[i].write(i32::MIN);
                        } else {
                            let s = score_move(
                                self.board,
                                m,
                                None,
                                [None, None],
                                self.counter_move,
                                history,
                                self.color,
                            );
                            self.scores[i].write(s);
                        }
                    }
                    self.phase = 5;
                }
                5 => {
                    // Phase 5: Yield quiet moves via swap-selection
                    let total_moves = self.moves.len();
                    while self.cur_quiet < total_moves {
                        let mut best_score = unsafe { self.scores[self.cur_quiet].assume_init() };
                        let mut best_idx = self.cur_quiet;

                        for i in (self.cur_quiet + 1)..total_moves {
                            let sc = unsafe { self.scores[i].assume_init() };
                            if sc > best_score {
                                best_score = sc;
                                best_idx = i;
                            }
                        }

                        if best_idx != self.cur_quiet {
                            self.moves.as_slice_mut().swap(self.cur_quiet, best_idx);
                            self.scores.swap(self.cur_quiet, best_idx);
                        }

                        let m = self.moves.as_slice()[self.cur_quiet];
                        self.cur_quiet += 1;

                        if best_score != i32::MIN {
                            return Some(m);
                        }
                    }
                    return None;
                }
                _ => return None,
            }
        }
    }
}
