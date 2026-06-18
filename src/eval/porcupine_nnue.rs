//! Custom Porcupine NNUE implementation (768 -> 128 -> 1)
//! 
//! Features: (side_to_move_idx * 6 * 64) + (piece_type * 64) + square_idx
//! Hidden: 128 (CReLU)
//! Output: 1 (Linear)

use crate::types::{Board, Color, Move, MoveFlag, Piece, Score, Square};
use std::sync::Arc;

pub const INPUT_SIZE: usize = 768;
pub const HIDDEN_SIZE: usize = 128;
pub const SCALE: f32 = 400.0;

/// The model weights
#[derive(Clone)]
pub struct Model {
    pub input_weights: Vec<f32>,   // [768 * 128]
    pub output_weights: Vec<f32>,  // [128]
    pub output_bias: f32,
}

impl Model {
    /// Load embedded weights from network.bin
    pub fn load_embedded() -> Arc<Self> {
        let bytes = include_bytes!("../../network.bin");
        
        let mut input_weights = vec![0.0f32; INPUT_SIZE * HIDDEN_SIZE];
        let mut output_weights = vec![0.0f32; HIDDEN_SIZE];
        let mut output_bias_arr = [0.0f32; 1];

        unsafe {
            let input_slice = std::slice::from_raw_parts_mut(
                input_weights.as_mut_ptr() as *mut u8, 
                input_weights.len() * 4
            );
            input_slice.copy_from_slice(&bytes[0..input_weights.len() * 4]);

            let offset1 = input_weights.len() * 4;
            let output_slice = std::slice::from_raw_parts_mut(
                output_weights.as_mut_ptr() as *mut u8, 
                output_weights.len() * 4
            );
            output_slice.copy_from_slice(&bytes[offset1..offset1 + output_weights.len() * 4]);

            let offset2 = offset1 + output_weights.len() * 4;
            let bias_slice = std::slice::from_raw_parts_mut(
                output_bias_arr.as_mut_ptr() as *mut u8, 
                4
            );
            bias_slice.copy_from_slice(&bytes[offset2..offset2 + 4]);
        }

        Arc::new(Self {
            input_weights,
            output_weights,
            output_bias: output_bias_arr[0],
        })
    }
}

/// A perspective-dependent accumulator
#[derive(Clone)]
pub struct Accumulator {
    pub vals: [f32; HIDDEN_SIZE],
}

impl Accumulator {
    pub fn new() -> Self {
        Self { vals: [0.0; HIDDEN_SIZE] }
    }

    #[inline(always)]
    pub fn add(&mut self, model: &Model, feature_idx: usize) {
        let offset = feature_idx * HIDDEN_SIZE;
        let weights = &model.input_weights[offset..offset + HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            self.vals[i] += weights[i];
        }
    }

    #[inline(always)]
    pub fn sub(&mut self, model: &Model, feature_idx: usize) {
        let offset = feature_idx * HIDDEN_SIZE;
        let weights = &model.input_weights[offset..offset + HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            self.vals[i] -= weights[i];
        }
    }

    #[inline(always)]
    pub fn evaluate(&self, model: &Model) -> f32 {
        let mut sum = model.output_bias;
        for i in 0..HIDDEN_SIZE {
            // Clipped ReLU: clamp(0, 1)
            let val = self.vals[i].max(0.0).min(1.0);
            sum += val * model.output_weights[i];
        }
        sum
    }
}

/// Feature mapping logic
#[inline(always)]
fn get_feature_index(stm: Color, piece_color: Color, piece: Piece, sq: Square) -> usize {
    let color_idx = if piece_color == stm { 0 } else { 1 };
    let pt = piece.index();
    let index_sq = if stm == Color::White {
        sq.index() as usize
    } else {
        sq.index() as usize ^ 56
    };
    (color_idx * 6 * 64) + (pt * 64) + index_sq
}

/// Stateful evaluator for Porcupine NNUE
#[derive(Clone)]
pub struct PorcupineEvaluator {
    model: Arc<Model>,
    // Stack of accumulators: [WhiteSTM, BlackSTM]
    states: Vec<[Accumulator; 2]>,
}

impl PorcupineEvaluator {
    pub fn new(model: Arc<Model>, board: &Board) -> Self {
        let mut states = Vec::with_capacity(128);
        let mut accs = [Accumulator::new(), Accumulator::new()];
        
        // Initial full calculation
        for sq in board.occupied() {
            let (piece, color) = board.piece_at(sq).unwrap();
            accs[0].add(&model, get_feature_index(Color::White, color, piece, sq));
            accs[1].add(&model, get_feature_index(Color::Black, color, piece, sq));
        }
        
        states.push(accs);
        Self { model, states }
    }

    #[inline]
    pub fn evaluate(&self, ply: usize, turn: Color) -> Score {
        let safe_ply = ply.min(self.states.len() - 1);
        let acc_idx = if turn == Color::White { 0 } else { 1 };
        let raw = self.states[safe_ply][acc_idx].evaluate(&self.model);
        Score::cp((raw * SCALE) as i32)
    }

    #[inline]
    pub fn update_move(&mut self, ply: usize, board: &Board, mv: Move) -> bool {
        let next_ply = ply + 1;
        while self.states.len() <= next_ply {
            self.states.push(self.states.last().unwrap().clone());
        }
        
        let prev_accs = self.states[ply].clone();
        let mut next_accs = prev_accs;
        
        let from = mv.from();
        let to = mv.to();
        let (piece, color) = board.piece_at(from).unwrap();
        let captured = board.piece_at(to);

        // Remove piece from old square
        for i in 0..2 {
            let stm = if i == 0 { Color::White } else { Color::Black };
            next_accs[i].sub(&self.model, get_feature_index(stm, color, piece, from));
        }

        // Handle capture
        if let Some((cap_piece, cap_color)) = captured {
            for i in 0..2 {
                let stm = if i == 0 { Color::White } else { Color::Black };
                next_accs[i].sub(&self.model, get_feature_index(stm, cap_color, cap_piece, to));
            }
        }

        // Handle en passant
        if piece == Piece::Pawn && mv.flag() == MoveFlag::EnPassant {
            let ep_sq = Square::from_file_rank(to.file(), from.rank());
            for i in 0..2 {
                let stm = if i == 0 { Color::White } else { Color::Black };
                next_accs[i].sub(&self.model, get_feature_index(stm, !color, Piece::Pawn, ep_sq));
            }
        }

        // Handle promotion
        let final_piece = if let Some(promo) = mv.flag().promotion_piece() {
            promo
        } else {
            piece
        };

        // Add piece to new square
        for i in 0..2 {
            let stm = if i == 0 { Color::White } else { Color::Black };
            next_accs[i].add(&self.model, get_feature_index(stm, color, final_piece, to));
        }

        // Handle castling (rook move)
        if mv.flag() == MoveFlag::KingCastle || mv.flag() == MoveFlag::QueenCastle {
            let (r_from, r_to) = if mv.flag() == MoveFlag::KingCastle {
                let rank = from.rank();
                (Square::from_file_rank(movegen::File::H, rank), Square::from_file_rank(movegen::File::F, rank))
            } else {
                let rank = from.rank();
                (Square::from_file_rank(movegen::File::A, rank), Square::from_file_rank(movegen::File::D, rank))
            };
            for i in 0..2 {
                let stm = if i == 0 { Color::White } else { Color::Black };
                next_accs[i].sub(&self.model, get_feature_index(stm, color, Piece::Rook, r_from));
                next_accs[i].add(&self.model, get_feature_index(stm, color, Piece::Rook, r_to));
            }
        }

        self.states[next_ply] = next_accs;
        true
    }

    #[inline]
    pub fn refresh(&mut self, ply: usize, board: &Board) {
        let mut accs = [Accumulator::new(), Accumulator::new()];
        for sq in board.occupied() {
            let (piece, color) = board.piece_at(sq).unwrap();
            accs[0].add(&self.model, get_feature_index(Color::White, color, piece, sq));
            accs[1].add(&self.model, get_feature_index(Color::Black, color, piece, sq));
        }
        if ply >= self.states.len() {
            self.states.resize(ply + 1, accs);
        } else {
            self.states[ply] = accs;
        }
    }
}
