//! LOD-NNUE v2.2 (Nano) — Normalized Symmetric HalfKP
//!
//! Input:  22,528 features  (32 king-buckets × 11 piece-types × 64 squares)
//! Layer 0 (Accumulator): 22,528 → 16  (per perspective, shared weights)
//! Dual perspective concat: [Us 16 | Them 16] = 32
//! Layer 1: 32 → 32   (Clipped ReLU)
//! Layer 2: 32 → 16   (Clipped ReLU)
//! Layer 3: 16 → 1    (Linear output → centipawn conversion)

use crate::types::{Board, Color, Move, MoveFlag, Piece, Score, Square};
use std::sync::Arc;

// ---- Architecture constants ----
pub const NUM_KING_BUCKETS: usize = 32;
pub const NUM_PIECE_TYPES: usize  = 11;  // 5 friendly + 6 enemy
pub const NUM_SQUARES: usize      = 64;
pub const INPUT_SIZE: usize       = NUM_KING_BUCKETS * NUM_PIECE_TYPES * NUM_SQUARES; // 22,528
pub const ACC_SIZE: usize         = 16;   // Per-perspective
pub const L1_SIZE: usize          = 32;   // ACC_SIZE * 2
pub const L2_SIZE: usize          = 32;
pub const L3_SIZE: usize          = 16;
pub const SCALE: f32              = 400.0;

// ========== Model ==========

#[derive(Clone)]
pub struct Model {
    // Accumulator: [22528 × 16]
    pub acc_weights: Vec<f32>,
    pub acc_bias: [f32; ACC_SIZE],
    // Layer 1: 32 → 32
    pub fc1_weights: Vec<f32>,   // [L2_SIZE × L1_SIZE] = [32 × 32]
    pub fc1_bias: [f32; L2_SIZE],
    // Layer 2: 32 → 16
    pub fc2_weights: Vec<f32>,   // [L3_SIZE × L2_SIZE] = [16 × 32]
    pub fc2_bias: [f32; L3_SIZE],
    // Layer 3: 16 → 1
    pub fc3_weights: [f32; L3_SIZE], // [1 × L3_SIZE] = [16]
    pub fc3_bias: f32,
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
        let bytes = include_bytes!("../../network.bin");

        let mut acc_weights = vec![0.0f32; INPUT_SIZE * ACC_SIZE];
        let mut acc_bias    = [0.0f32; ACC_SIZE];
        let mut fc1_weights = vec![0.0f32; L2_SIZE * L1_SIZE];
        let mut fc1_bias    = [0.0f32; L2_SIZE];
        let mut fc2_weights = vec![0.0f32; L3_SIZE * L2_SIZE];
        let mut fc2_bias    = [0.0f32; L3_SIZE];
        let mut fc3_weights = [0.0f32; L3_SIZE];
        let mut fc3_bias_arr = [0.0f32; 1];

        unsafe {
            let mut off = 0usize;
            read_f32_slice(bytes, &mut off, &mut acc_weights);
            read_f32_slice(bytes, &mut off, &mut acc_bias);
            read_f32_slice(bytes, &mut off, &mut fc1_weights);
            read_f32_slice(bytes, &mut off, &mut fc1_bias);
            read_f32_slice(bytes, &mut off, &mut fc2_weights);
            read_f32_slice(bytes, &mut off, &mut fc2_bias);
            read_f32_slice(bytes, &mut off, &mut fc3_weights);
            read_f32_slice(bytes, &mut off, &mut fc3_bias_arr);
        }

        Arc::new(Self {
            acc_weights,
            acc_bias,
            fc1_weights,
            fc1_bias,
            fc2_weights,
            fc2_bias,
            fc3_weights,
            fc3_bias: fc3_bias_arr[0],
        })
    }
}

// ========== Normalization helpers ==========

/// Horizontal mirror: files a↔h, b↔g, c↔f, d↔e
#[inline(always)]
fn mirror_h(sq: Square) -> Square {
    Square::from_index(sq.index() ^ 7).unwrap()
}

/// Vertical flip: rank 1↔8, rank 2↔7, etc.
#[inline(always)]
fn flip_v(sq: Square) -> Square {
    Square::from_index(sq.index() ^ 56).unwrap()
}

/// Does the king (on a raw or vertically-flipped square) sit on files e-h?
#[inline(always)]
fn needs_mirror(sq: Square) -> bool {
    sq.file().index() >= 4
}

/// King bucket (0..31) from a *normalized* square (files a-d guaranteed).
#[inline(always)]
fn king_bucket(sq: Square) -> usize {
    let file = sq.file().index() as usize;
    let rank = sq.rank().index() as usize;
    rank * 4 + file
}

/// Normalize a square for a given perspective.
/// 1) If perspective is Black, flip vertically.
/// 2) If `do_mirror`, mirror horizontally.
#[inline(always)]
fn normalize_sq(sq: Square, perspective: Color, do_mirror: bool) -> Square {
    let mut s = sq;
    if perspective == Color::Black {
        s = flip_v(s);
    }
    if do_mirror {
        s = mirror_h(s);
    }
    s
}

// ========== Feature index ==========

/// Compute a single HalfKP feature index.
///
/// `bucket`:     king bucket (0..31) for this perspective
/// `piece`:      the piece type on the square
/// `piece_color`:the color of that piece
/// `perspective`:the color whose perspective we are computing
/// `norm_sq`:    the already-normalized square index (0..63)
#[inline(always)]
fn halfkp_feature(bucket: usize, piece: Piece, piece_color: Color,
                  perspective: Color, norm_sq: usize) -> usize {
    let is_friendly = piece_color == perspective;
    let pt = if is_friendly {
        piece.index()          // P=0 N=1 B=2 R=3 Q=4
    } else {
        piece.index() + 5      // P=5 N=6 B=7 R=8 Q=9 K=10
    };
    bucket * (NUM_PIECE_TYPES * NUM_SQUARES) + pt * NUM_SQUARES + norm_sq
}

// ========== Accumulator ==========

#[derive(Clone)]
pub struct Accumulator {
    pub vals: [f32; ACC_SIZE],
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
        for i in 0..ACC_SIZE { self.vals[i] += w[i]; }
    }

    #[inline(always)]
    pub fn sub(&mut self, model: &Model, feature_idx: usize) {
        let off = feature_idx * ACC_SIZE;
        let w = &model.acc_weights[off..off + ACC_SIZE];
        for i in 0..ACC_SIZE { self.vals[i] -= w[i]; }
    }
}

// ========== Forward pass (from concatenated accumulators) ==========

/// Run layers 1-3 on a [Us(16) ++ Them(16)] = 32 vector.
#[inline]
fn forward_pass(model: &Model, stm: &[f32; ACC_SIZE], nstm: &[f32; ACC_SIZE]) -> f32 {
    // Clipped-ReLU the accumulator outputs
    let mut input = [0.0f32; L1_SIZE]; // 32
    for i in 0..ACC_SIZE {
        input[i]            = stm[i].max(0.0).min(1.0);
        input[ACC_SIZE + i] = nstm[i].max(0.0).min(1.0);
    }

    // Layer 1:  32 → 32,  CReLU
    let mut l1 = [0.0f32; L2_SIZE];
    for i in 0..L2_SIZE {
        let mut sum = model.fc1_bias[i];
        let off = i * L1_SIZE;
        for j in 0..L1_SIZE {
            sum += model.fc1_weights[off + j] * input[j];
        }
        l1[i] = sum.max(0.0).min(1.0);
    }

    // Layer 2:  32 → 16,  CReLU
    let mut l2 = [0.0f32; L3_SIZE];
    for i in 0..L3_SIZE {
        let mut sum = model.fc2_bias[i];
        let off = i * L2_SIZE;
        for j in 0..L2_SIZE {
            sum += model.fc2_weights[off + j] * l1[j];
        }
        l2[i] = sum.max(0.0).min(1.0);
    }

    // Layer 3:  16 → 1
    let mut out = model.fc3_bias;
    for i in 0..L3_SIZE {
        out += model.fc3_weights[i] * l2[i];
    }
    out
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
            let king_sq_raw = board.king_square(perspective);

            let mut ksq = king_sq_raw;
            if perspective == Color::Black { ksq = flip_v(ksq); }
            let do_mirror = needs_mirror(ksq);
            if do_mirror { ksq = mirror_h(ksq); }
            let bucket = king_bucket(ksq);

            for sq in board.occupied() {
                let (piece, color) = board.piece_at(sq).unwrap();
                // Skip this perspective's own King
                if piece == Piece::King && color == perspective { continue; }

                let norm_sq = normalize_sq(sq, perspective, do_mirror).index() as usize;
                let idx = halfkp_feature(bucket, piece, color, perspective, norm_sq);
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

        let raw = forward_pass(
            &self.model,
            &self.states[safe_ply].accs[stm_idx].vals,
            &self.states[safe_ply].accs[nstm_idx].vals,
        );

        // Convert logit → centipawns:  cp = raw × SCALE / ln(10)
        Score::cp((raw * 173.7178) as i32)
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

        // King move → need full refresh (king position changes bucket/mirror)
        // Return false so the caller invokes refresh() with the post-move board.
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
            let king_sq_raw = board.king_square(perspective);

            let mut ksq = king_sq_raw;
            if perspective == Color::Black { ksq = flip_v(ksq); }
            let do_mirror = needs_mirror(ksq);
            if do_mirror { ksq = mirror_h(ksq); }
            let bucket = king_bucket(ksq);

            // Remove piece from old square
            let from_norm = normalize_sq(from, perspective, do_mirror).index() as usize;
            let from_idx = halfkp_feature(bucket, piece, color, perspective, from_norm);
            self.states[next_ply].accs[perspective_idx].sub(&self.model, from_idx);

            // Handle capture
            if let Some((cap_piece, cap_color)) = captured {
                let to_norm = normalize_sq(to, perspective, do_mirror).index() as usize;
                let cap_idx = halfkp_feature(bucket, cap_piece, cap_color, perspective, to_norm);
                self.states[next_ply].accs[perspective_idx].sub(&self.model, cap_idx);
            }

            // Handle en passant
            if piece == Piece::Pawn && mv.flag() == MoveFlag::EnPassant {
                let ep_sq = Square::from_file_rank(to.file(), from.rank());
                let ep_norm = normalize_sq(ep_sq, perspective, do_mirror).index() as usize;
                let ep_idx = halfkp_feature(bucket, Piece::Pawn, !color, perspective, ep_norm);
                self.states[next_ply].accs[perspective_idx].sub(&self.model, ep_idx);
            }

            // Determine final piece (promotion)
            let final_piece = if let Some(promo) = mv.flag().promotion_piece() {
                promo
            } else {
                piece
            };

            // Add piece to new square
            let to_norm = normalize_sq(to, perspective, do_mirror).index() as usize;
            let to_idx = halfkp_feature(bucket, final_piece, color, perspective, to_norm);
            self.states[next_ply].accs[perspective_idx].add(&self.model, to_idx);

            // Handle castling rook
            if mv.flag() == MoveFlag::KingCastle || mv.flag() == MoveFlag::QueenCastle {
                let (r_from, r_to) = if mv.flag() == MoveFlag::KingCastle {
                    let rank = from.rank();
                    (Square::from_file_rank(movegen::File::H, rank),
                     Square::from_file_rank(movegen::File::F, rank))
                } else {
                    let rank = from.rank();
                    (Square::from_file_rank(movegen::File::A, rank),
                     Square::from_file_rank(movegen::File::D, rank))
                };
                let rf_norm = normalize_sq(r_from, perspective, do_mirror).index() as usize;
                let rt_norm = normalize_sq(r_to, perspective, do_mirror).index() as usize;
                let rf_idx = halfkp_feature(bucket, Piece::Rook, color, perspective, rf_norm);
                let rt_idx = halfkp_feature(bucket, Piece::Rook, color, perspective, rt_norm);
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
        eval.update_move(0, &board, mv);
        board.make_move(mv);

        let inc_eval = eval.evaluate(1, Color::Black);

        let mut eval_scratch = PorcupineEvaluator::new(model, &board);
        let scratch_eval = eval_scratch.evaluate(0, Color::Black);

        assert_eq!(inc_eval.0, scratch_eval.0,
                   "Incremental eval mismatch! inc: {}, scratch: {}",
                   inc_eval.0, scratch_eval.0);
    }
}
