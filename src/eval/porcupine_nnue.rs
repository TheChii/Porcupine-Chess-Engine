//! Porcupine NNUE — HalfKP 256 → 32 → 32 → 1
//!
//! Input:  49152 features (64 King buckets × 768 features)
//! Layer 0 (Accumulator): 49152 → 256  (per perspective)
//! Dual perspective concat: [STM 256 | NSTM 256] = 512
//! Layer 1: 512 → 32 (CReLU)
//! Layer 2: 32 → 32 (CReLU)
//! Layer 3: 32 → 1
//!
//! Output scaling: 400.0

use crate::types::{Board, Color, Move, MoveFlag, Piece, Score, Square};
use std::sync::Arc;

// ---- Architecture constants ----
pub const NUM_PIECE_TYPES: usize  = 12;  // 6 friendly + 6 enemy
pub const NUM_SQUARES: usize      = 64;
pub const INPUT_SIZE: usize       = NUM_SQUARES * NUM_PIECE_TYPES * NUM_SQUARES; // 49152
pub const ACC_SIZE: usize         = 256;   // Per-perspective

// ---- Quantization constants ----
const QA: f32    = 255.0; // Layer 0 quantization scale
const SCALE: f32 = 400.0; // Eval scale (sigmoid → centipawns)

// ========== Model ==========

#[derive(Clone)]
pub struct Model {
    // Accumulator: [49152 × 128]
    pub acc_weights: Vec<i16>,
    pub acc_bias: [i16; ACC_SIZE],
    
    // Hidden Layer 1: 512 → 32
    pub l1_weights: [f32; 512 * 32],
    pub l1_bias: [f32; 32],
    
    // Hidden Layer 2: 32 → 32
    pub l2_weights: [f32; 32 * 32],
    pub l2_bias: [f32; 32],
    
    // Output Layer: 32 → 1
    pub l3_weights: [f32; 32],
    pub l3_bias: f32,
}

/// Helper to read a `&[i16]` from a byte slice at a given offset
unsafe fn read_i16_slice(bytes: &[u8], offset: &mut usize, out: &mut [i16]) {
    let len = out.len() * 2;
    let ptr = out.as_mut_ptr() as *mut u8;
    std::ptr::copy_nonoverlapping(bytes[*offset..].as_ptr(), ptr, len);
    *offset += len;
}

/// Helper to read a `&[f32]` from a byte slice at a given offset
unsafe fn read_f32_slice(bytes: &[u8], offset: &mut usize, out: &mut [f32]) {
    let len = out.len() * 4;
    let ptr = out.as_mut_ptr() as *mut u8;
    std::ptr::copy_nonoverlapping(bytes[*offset..].as_ptr(), ptr, len);
    *offset += len;
}

impl Model {
    pub fn load_embedded() -> Arc<Self> {
        let bytes = include_bytes!("../../quantised.bin");

        let mut acc_weights = vec![0i16; INPUT_SIZE * ACC_SIZE];
        let mut acc_bias    = [0i16; ACC_SIZE];
        let mut l1_weights  = [0.0f32; 512 * 32];
        let mut l1_bias     = [0.0f32; 32];
        let mut l2_weights  = [0.0f32; 32 * 32];
        let mut l2_bias     = [0.0f32; 32];
        let mut l3_weights  = [0.0f32; 32];
        let mut l3_bias_arr = [0.0f32; 1];

        unsafe {
            let mut off = 0usize;
            read_i16_slice(bytes, &mut off, &mut acc_weights);
            read_i16_slice(bytes, &mut off, &mut acc_bias);
            read_f32_slice(bytes, &mut off, &mut l1_weights);
            read_f32_slice(bytes, &mut off, &mut l1_bias);
            read_f32_slice(bytes, &mut off, &mut l2_weights);
            read_f32_slice(bytes, &mut off, &mut l2_bias);
            read_f32_slice(bytes, &mut off, &mut l3_weights);
            read_f32_slice(bytes, &mut off, &mut l3_bias_arr);
        }

        Arc::new(Self {
            acc_weights,
            acc_bias,
            l1_weights,
            l1_bias,
            l2_weights,
            l2_bias,
            l3_weights,
            l3_bias: l3_bias_arr[0],
        })
    }
}

// ========== Normalization helpers ==========

/// Vertical flip: rank 1↔8, rank 2↔7, etc.
#[inline(always)]
fn flip_v(sq: Square) -> Square {
    Square::from_index(sq.index() ^ 56).unwrap()
}

/// Normalize a square for a given perspective.
/// 1) If perspective is Black, flip vertically.
#[inline(always)]
fn normalize_sq(sq: Square, perspective: Color) -> Square {
    let mut s = sq;
    if perspective == Color::Black {
        s = flip_v(s);
    }
    s
}

// ========== Feature index ==========

/// Compute a single feature index for HalfKP.
/// Maps to 64 king buckets × 768 standard piece features.
#[inline(always)]
fn halfkp_feature(piece: Piece, piece_color: Color,
                  perspective: Color, norm_sq: usize, king_sq_norm: usize) -> usize {
    let is_friendly = piece_color == perspective;
    let pt = if is_friendly {
        piece.index()          // P=0 N=1 B=2 R=3 Q=4 K=5
    } else {
        piece.index() + 6      // P=6 N=7 B=8 R=9 Q=10 K=11
    };
    let piece_feat = pt * NUM_SQUARES + norm_sq;
    king_sq_norm * 768 + piece_feat
}

// ========== Accumulator ==========

#[derive(Clone)]
pub struct Accumulator {
    pub vals: [i16; ACC_SIZE],
}

impl Accumulator {
    #[inline]
    pub fn from_bias(model: &Model) -> Self {
        Self { vals: model.acc_bias }
    }

    #[inline(always)]
    pub fn add(&mut self, model: &Model, feature_idx: usize) {
        let off = feature_idx * ACC_SIZE;
        let w = &model.acc_weights[off..off + ACC_SIZE];
        for i in 0..ACC_SIZE { self.vals[i] = self.vals[i].wrapping_add(w[i]); }
    }

    #[inline(always)]
    pub fn sub(&mut self, model: &Model, feature_idx: usize) {
        let off = feature_idx * ACC_SIZE;
        let w = &model.acc_weights[off..off + ACC_SIZE];
        for i in 0..ACC_SIZE { self.vals[i] = self.vals[i].wrapping_sub(w[i]); }
    }
}

// ========== Forward pass ==========

/// Run hidden layers on [STM(256) ++ NSTM(256)] = 512 vector using unquantized f32 weights.
#[inline]
fn forward_pass(model: &Model, stm: &[i16; ACC_SIZE], nstm: &[i16; ACC_SIZE]) -> i32 {
    // 1. CReLU and scale to floats for HL1
    let mut hl1 = [0.0f32; 512];
    for i in 0..ACC_SIZE {
        hl1[i] = stm[i].clamp(0, QA as i16) as f32 / QA;
        hl1[ACC_SIZE + i] = nstm[i].clamp(0, QA as i16) as f32 / QA;
    }

    // 2. HL2 (512 -> 32)
    let mut hl2 = model.l1_bias;
    for in_idx in 0..512 {
        let val = hl1[in_idx];
        for out_idx in 0..32 {
            hl2[out_idx] += val * model.l1_weights[in_idx * 32 + out_idx];
        }
    }
    for i in 0..32 { hl2[i] = hl2[i].clamp(0.0, 1.0); } // CReLU

    // 3. HL3 (32 -> 32)
    let mut hl3 = model.l2_bias;
    for in_idx in 0..32 {
        let val = hl2[in_idx];
        for out_idx in 0..32 {
            hl3[out_idx] += val * model.l2_weights[in_idx * 32 + out_idx];
        }
    }
    for i in 0..32 { hl3[i] = hl3[i].clamp(0.0, 1.0); } // CReLU

    // 4. Output (32 -> 1)
    let mut out = model.l3_bias;
    for in_idx in 0..32 {
        out += hl3[in_idx] * model.l3_weights[in_idx];
    }

    // 5. Apply eval scale -> centipawns
    (out * SCALE) as i32
}

// ========== Per-ply state ==========

/// Holds both perspective accumulators.
#[derive(Clone)]
struct PlyState {
    accs: [Accumulator; 2],   // [0] = White perspective, [1] = Black perspective
}

// ========== Evaluator ==========

#[derive(Clone)]
pub struct PorcupineEvaluator {
    model: Arc<Model>,
    states: Vec<PlyState>,
}

impl PorcupineEvaluator {
    /// Build accumulators for the initial position.
    pub fn new(model: Arc<Model>, board: &Board) -> Self {
        let state = Self::compute_full(&model, board);
        let mut states = Vec::with_capacity(128);
        states.push(state);
        Self { model, states }
    }

    /// Full (non-incremental) accumulator computation for both perspectives.
    fn compute_full(model: &Model, board: &Board) -> PlyState {
        let mut accs = [Accumulator::from_bias(model), Accumulator::from_bias(model)];

        for perspective_idx in 0..2 {
            let perspective = if perspective_idx == 0 { Color::White } else { Color::Black };
            let k_sq = board.king_square(perspective);
            let k_sq_norm = normalize_sq(k_sq, perspective).index() as usize;

            for sq in board.occupied() {
                let (piece, color) = board.piece_at(sq).unwrap();

                let norm_sq = normalize_sq(sq, perspective).index() as usize;
                let idx = halfkp_feature(piece, color, perspective, norm_sq, k_sq_norm);
                accs[perspective_idx].add(model, idx);
            }
        }

        PlyState { accs }
    }

    // ---- Evaluate ----

    #[inline]
    pub fn evaluate(&mut self, ply: usize, turn: Color) -> Score {
        let safe_ply = ply.min(self.states.len() - 1);

        let stm_idx  = if turn == Color::White { 0 } else { 1 };
        let nstm_idx = 1 - stm_idx;

        let cp = forward_pass(
            &self.model,
            &self.states[safe_ply].accs[stm_idx].vals,
            &self.states[safe_ply].accs[nstm_idx].vals,
        );

        let clamped = cp.clamp(-20000, 20000);
        Score::cp(clamped)
    }

    // ---- Incremental update ----

    #[inline]
    pub fn update_move(&mut self, ply: usize, board: &Board, mv: Move) -> bool {
        let next_ply = ply + 1;
        while self.states.len() <= next_ply {
            self.states.push(self.states.last().unwrap().clone());
        }

        let from = mv.from();
        let to   = mv.to();
        let (piece, color) = board.piece_at(from).unwrap();

        // If a king moves, the bucket for that perspective changes, requiring a full refresh.
        // Returning `false` tells the searcher to call `refresh()` automatically.
        if piece == Piece::King {
            return false;
        }

        // Clone from parent ply
        let prev = self.states[ply].clone();
        self.states[next_ply] = prev;

        let captured = board.piece_at(to);

        // Incremental update for both perspectives
        for perspective_idx in 0..2 {
            let perspective = if perspective_idx == 0 { Color::White } else { Color::Black };
            let k_sq = board.king_square(perspective);
            let k_sq_norm = normalize_sq(k_sq, perspective).index() as usize;

            // Remove piece from old square
            let from_norm = normalize_sq(from, perspective).index() as usize;
            let from_idx = halfkp_feature(piece, color, perspective, from_norm, k_sq_norm);
            self.states[next_ply].accs[perspective_idx].sub(&self.model, from_idx);

            // Handle capture
            if let Some((cap_piece, cap_color)) = captured {
                let to_norm = normalize_sq(to, perspective).index() as usize;
                let cap_idx = halfkp_feature(cap_piece, cap_color, perspective, to_norm, k_sq_norm);
                self.states[next_ply].accs[perspective_idx].sub(&self.model, cap_idx);
            }

            // Handle en passant
            if piece == Piece::Pawn && mv.flag() == MoveFlag::EnPassant {
                // The captured pawn is on the same rank as the from square
                let ep_sq = Square::from_file_rank(to.file(), from.rank());
                let ep_norm = normalize_sq(ep_sq, perspective).index() as usize;
                let ep_idx = halfkp_feature(Piece::Pawn, !color, perspective, ep_norm, k_sq_norm);
                self.states[next_ply].accs[perspective_idx].sub(&self.model, ep_idx);
            }

            // Determine final piece (promotion)
            let final_piece = if let Some(promo) = mv.flag().promotion_piece() {
                promo
            } else {
                piece
            };

            // Add piece to new square
            let to_norm = normalize_sq(to, perspective).index() as usize;
            let to_idx = halfkp_feature(final_piece, color, perspective, to_norm, k_sq_norm);
            self.states[next_ply].accs[perspective_idx].add(&self.model, to_idx);

            // Handle castling rook (Note: since King moves trigger a full refresh, this block 
            // is technically unreachable because `piece == Piece::King` caught it, but we keep it
            // for completeness in case castling is ever represented as a rook move natively).
            if mv.flag() == MoveFlag::KingCastle || mv.flag() == MoveFlag::QueenCastle {
                let (r_from, r_to) = if mv.flag() == MoveFlag::KingCastle {
                    let rank = from.rank();
                    (Square::from_index(rank.index() as u8 * 8 + 7).unwrap(),
                     Square::from_index(rank.index() as u8 * 8 + 5).unwrap())
                } else {
                    let rank = from.rank();
                    (Square::from_index(rank.index() as u8 * 8).unwrap(),
                     Square::from_index(rank.index() as u8 * 8 + 3).unwrap())
                };
                let rf_norm = normalize_sq(r_from, perspective).index() as usize;
                let rt_norm = normalize_sq(r_to, perspective).index() as usize;
                let rf_idx = halfkp_feature(Piece::Rook, color, perspective, rf_norm, k_sq_norm);
                let rt_idx = halfkp_feature(Piece::Rook, color, perspective, rt_norm, k_sq_norm);
                self.states[next_ply].accs[perspective_idx].sub(&self.model, rf_idx);
                self.states[next_ply].accs[perspective_idx].add(&self.model, rt_idx);
            }
        }

        true
    }

    // ---- Full refresh ----

    #[inline]
    pub fn refresh(&mut self, ply: usize, board: &Board) {
        let state = Self::compute_full(&self.model, board);
        if ply >= self.states.len() {
            self.states.resize(ply + 1, state);
        } else {
            self.states[ply] = state;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnue_evaluate() {
        let model = Model::load_embedded();
        let board = Board::default();
        let mut eval = PorcupineEvaluator::new(model, &board);
        let score = eval.evaluate(0, Color::White);
        assert!(score.0 >= -4000 && score.0 <= 4000,
                "Starting-position score out of range: {}", score.0);
    }

    #[test]
    fn test_nnue_incremental_update() {
        let model = Model::load_embedded();
        let mut board = Board::default();
        let mut eval = PorcupineEvaluator::new(model.clone(), &board);

        // Play e2e4 (pawn, not a king move → incremental)
        let mv = Move::new(Square::E2, Square::E4, MoveFlag::Quiet);
        assert_eq!(eval.update_move(0, &board, mv), true);
        board.make_move(mv);

        let inc_eval = eval.evaluate(1, Color::Black);

        let mut eval_scratch = PorcupineEvaluator::new(model.clone(), &board);
        let scratch_eval = eval_scratch.evaluate(0, Color::Black);

        assert_eq!(inc_eval.0, scratch_eval.0,
                   "Incremental eval mismatch! inc: {}, scratch: {}",
                   inc_eval.0, scratch_eval.0);
                   
        // Test King move triggers refresh
        let mut board = Board::default();
        let mv = Move::new(Square::E1, Square::E2, MoveFlag::Quiet); // Dummy king move
        assert_eq!(eval.update_move(0, &board, mv), false);
    }
}
