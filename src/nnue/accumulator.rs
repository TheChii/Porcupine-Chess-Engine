//! NNUE accumulators: perspective state and incremental updates.
//!
//! Maintains HalfKAv2 accumulators (512-dim signed 16-bit integers) and
//! optional PSQT skip accumulators (8-channel signed 32-bit integers)
//! for both White and Black perspectives.
//!
//! # Correctness Invariant
//!
//! Each perspective's accumulator is valid if and only if its king has not changed
//! canonical bucket. If a perspective's king moves, that perspective's accumulator
//! MUST be refreshed from scratch. The non-moving king's perspective can always be
//! incrementally updated.

use super::features::{
    compute_feature_diff_non_king, enumerate_features, FeatureIndex, PerspectiveOrientation,
};
use crate::types::{Board, Color, Move, MoveFlag, Piece, Square};

/// Width of the feature transformer accumulator.
pub const TRANSFORMER_WIDTH: usize = 512;

/// Number of PSQT skip accumulator channels.
pub const PSQT_CHANNELS: usize = 8;

/// Single perspective accumulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accumulator {
    pub vals: [i16; TRANSFORMER_WIDTH],
    pub psqt: [i32; PSQT_CHANNELS],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            vals: [0; TRANSFORMER_WIDTH],
            psqt: [0; PSQT_CHANNELS],
        }
    }
}

impl Accumulator {
    /// Initialize accumulator from biases.
    #[inline]
    pub fn new(bias: &[i16; TRANSFORMER_WIDTH], psqt_bias: &[i32; PSQT_CHANNELS]) -> Self {
        Self {
            vals: *bias,
            psqt: *psqt_bias,
        }
    }

    /// Add a feature vector to the accumulator.
    #[inline(always)]
    pub fn add_feature(
        &mut self,
        feat: FeatureIndex,
        weights: &[[i16; TRANSFORMER_WIDTH]],
        psqt_weights: Option<&[[i32; PSQT_CHANNELS]]>,
    ) {
        debug_assert!(feat < weights.len());
        let w = &weights[feat];

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe {
                vec_add_512_avx2(&mut self.vals, w);
            }
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    vec_add_512_avx2(&mut self.vals, w);
                }
            } else {
                for i in 0..TRANSFORMER_WIDTH {
                    self.vals[i] = self.vals[i].saturating_add(w[i]);
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..TRANSFORMER_WIDTH {
                self.vals[i] = self.vals[i].saturating_add(w[i]);
            }
        }

        if let Some(pw) = psqt_weights {
            let p = &pw[feat];
            for i in 0..PSQT_CHANNELS {
                self.psqt[i] = self.psqt[i].saturating_add(p[i]);
            }
        }
    }

    /// Subtract a feature vector from the accumulator.
    #[inline(always)]
    pub fn sub_feature(
        &mut self,
        feat: FeatureIndex,
        weights: &[[i16; TRANSFORMER_WIDTH]],
        psqt_weights: Option<&[[i32; PSQT_CHANNELS]]>,
    ) {
        debug_assert!(feat < weights.len());
        let w = &weights[feat];

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe {
                vec_sub_512_avx2(&mut self.vals, w);
            }
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    vec_sub_512_avx2(&mut self.vals, w);
                }
            } else {
                for i in 0..TRANSFORMER_WIDTH {
                    self.vals[i] = self.vals[i].saturating_sub(w[i]);
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..TRANSFORMER_WIDTH {
                self.vals[i] = self.vals[i].saturating_sub(w[i]);
            }
        }

        if let Some(pw) = psqt_weights {
            let p = &pw[feat];
            for i in 0..PSQT_CHANNELS {
                self.psqt[i] = self.psqt[i].saturating_sub(p[i]);
            }
        }
    }
}

/// AVX2 512-element saturating vector addition (unrolled 4x).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn vec_add_512_avx2(acc: &mut [i16; TRANSFORMER_WIDTH], w: &[i16; TRANSFORMER_WIDTH]) {
    use core::arch::x86_64::*;
    let mut a_ptr = acc.as_mut_ptr() as *mut __m256i;
    let mut w_ptr = w.as_ptr() as *const __m256i;
    for _ in 0..8 {
        let a0 = _mm256_loadu_si256(a_ptr);
        let a1 = _mm256_loadu_si256(a_ptr.add(1));
        let a2 = _mm256_loadu_si256(a_ptr.add(2));
        let a3 = _mm256_loadu_si256(a_ptr.add(3));
        let w0 = _mm256_loadu_si256(w_ptr);
        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let w3 = _mm256_loadu_si256(w_ptr.add(3));
        _mm256_storeu_si256(a_ptr, _mm256_adds_epi16(a0, w0));
        _mm256_storeu_si256(a_ptr.add(1), _mm256_adds_epi16(a1, w1));
        _mm256_storeu_si256(a_ptr.add(2), _mm256_adds_epi16(a2, w2));
        _mm256_storeu_si256(a_ptr.add(3), _mm256_adds_epi16(a3, w3));
        a_ptr = a_ptr.add(4);
        w_ptr = w_ptr.add(4);
    }
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_add_512_avx2(acc: &mut [i16; TRANSFORMER_WIDTH], w: &[i16; TRANSFORMER_WIDTH]) {
    use core::arch::x86_64::*;
    let mut a_ptr = acc.as_mut_ptr() as *mut __m256i;
    let mut w_ptr = w.as_ptr() as *const __m256i;
    for _ in 0..8 {
        let a0 = _mm256_loadu_si256(a_ptr);
        let a1 = _mm256_loadu_si256(a_ptr.add(1));
        let a2 = _mm256_loadu_si256(a_ptr.add(2));
        let a3 = _mm256_loadu_si256(a_ptr.add(3));
        let w0 = _mm256_loadu_si256(w_ptr);
        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let w3 = _mm256_loadu_si256(w_ptr.add(3));
        _mm256_storeu_si256(a_ptr, _mm256_adds_epi16(a0, w0));
        _mm256_storeu_si256(a_ptr.add(1), _mm256_adds_epi16(a1, w1));
        _mm256_storeu_si256(a_ptr.add(2), _mm256_adds_epi16(a2, w2));
        _mm256_storeu_si256(a_ptr.add(3), _mm256_adds_epi16(a3, w3));
        a_ptr = a_ptr.add(4);
        w_ptr = w_ptr.add(4);
    }
}

/// AVX2 512-element saturating vector subtraction (unrolled 4x).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn vec_sub_512_avx2(acc: &mut [i16; TRANSFORMER_WIDTH], w: &[i16; TRANSFORMER_WIDTH]) {
    use core::arch::x86_64::*;
    let mut a_ptr = acc.as_mut_ptr() as *mut __m256i;
    let mut w_ptr = w.as_ptr() as *const __m256i;
    for _ in 0..8 {
        let a0 = _mm256_loadu_si256(a_ptr);
        let a1 = _mm256_loadu_si256(a_ptr.add(1));
        let a2 = _mm256_loadu_si256(a_ptr.add(2));
        let a3 = _mm256_loadu_si256(a_ptr.add(3));
        let w0 = _mm256_loadu_si256(w_ptr);
        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let w3 = _mm256_loadu_si256(w_ptr.add(3));
        _mm256_storeu_si256(a_ptr, _mm256_subs_epi16(a0, w0));
        _mm256_storeu_si256(a_ptr.add(1), _mm256_subs_epi16(a1, w1));
        _mm256_storeu_si256(a_ptr.add(2), _mm256_subs_epi16(a2, w2));
        _mm256_storeu_si256(a_ptr.add(3), _mm256_subs_epi16(a3, w3));
        a_ptr = a_ptr.add(4);
        w_ptr = w_ptr.add(4);
    }
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_sub_512_avx2(acc: &mut [i16; TRANSFORMER_WIDTH], w: &[i16; TRANSFORMER_WIDTH]) {
    use core::arch::x86_64::*;
    let mut a_ptr = acc.as_mut_ptr() as *mut __m256i;
    let mut w_ptr = w.as_ptr() as *const __m256i;
    for _ in 0..8 {
        let a0 = _mm256_loadu_si256(a_ptr);
        let a1 = _mm256_loadu_si256(a_ptr.add(1));
        let a2 = _mm256_loadu_si256(a_ptr.add(2));
        let a3 = _mm256_loadu_si256(a_ptr.add(3));
        let w0 = _mm256_loadu_si256(w_ptr);
        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let w3 = _mm256_loadu_si256(w_ptr.add(3));
        _mm256_storeu_si256(a_ptr, _mm256_subs_epi16(a0, w0));
        _mm256_storeu_si256(a_ptr.add(1), _mm256_subs_epi16(a1, w1));
        _mm256_storeu_si256(a_ptr.add(2), _mm256_subs_epi16(a2, w2));
        _mm256_storeu_si256(a_ptr.add(3), _mm256_subs_epi16(a3, w3));
        a_ptr = a_ptr.add(4);
        w_ptr = w_ptr.add(4);
    }
}

/// Perspective accumulators holding White and Black states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerspectiveAccumulators {
    pub white: Accumulator,
    pub black: Accumulator,
    pub white_orient: PerspectiveOrientation,
    pub black_orient: PerspectiveOrientation,
}

impl Default for PerspectiveAccumulators {
    fn default() -> Self {
        Self {
            white: Accumulator::default(),
            black: Accumulator::default(),
            white_orient: PerspectiveOrientation::default(),
            black_orient: PerspectiveOrientation::default(),
        }
    }
}

impl PerspectiveAccumulators {
    /// Full refresh of both perspectives from scratch for a given board.
    pub fn refresh(
        board: &Board,
        bias: &[i16; TRANSFORMER_WIDTH],
        weights: &[[i16; TRANSFORMER_WIDTH]],
        psqt_bias: &[i32; PSQT_CHANNELS],
        psqt_weights: Option<&[[i32; PSQT_CHANNELS]]>,
    ) -> Self {
        let white_orient = PerspectiveOrientation::new(board, Color::White);
        let black_orient = PerspectiveOrientation::new(board, Color::Black);

        let mut white_acc = Accumulator::new(bias, psqt_bias);
        let mut black_acc = Accumulator::new(bias, psqt_bias);

        let w_features = enumerate_features(board, Color::White);
        for &f in &w_features {
            white_acc.add_feature(f, weights, psqt_weights);
        }

        let b_features = enumerate_features(board, Color::Black);
        for &f in &b_features {
            black_acc.add_feature(f, weights, psqt_weights);
        }

        Self {
            white: white_acc,
            black: black_acc,
            white_orient,
            black_orient,
        }
    }

    /// Refresh a single perspective from scratch.
    pub fn refresh_perspective(
        board: &Board,
        perspective: Color,
        bias: &[i16; TRANSFORMER_WIDTH],
        weights: &[[i16; TRANSFORMER_WIDTH]],
        psqt_bias: &[i32; PSQT_CHANNELS],
        psqt_weights: Option<&[[i32; PSQT_CHANNELS]]>,
    ) -> (Accumulator, PerspectiveOrientation) {
        let orient = PerspectiveOrientation::new(board, perspective);
        let mut acc = Accumulator::new(bias, psqt_bias);
        let features = enumerate_features(board, perspective);
        for &f in &features {
            acc.add_feature(f, weights, psqt_weights);
        }
        (acc, orient)
    }

    /// Incrementally update accumulators across a move.
    ///
    /// `board`: board state BEFORE the move.
    /// `new_board`: board state AFTER the move.
    /// `mv`: the move played.
    pub fn update_move(
        &self,
        board: &Board,
        new_board: &Board,
        mv: Move,
        bias: &[i16; TRANSFORMER_WIDTH],
        weights: &[[i16; TRANSFORMER_WIDTH]],
        psqt_bias: &[i32; PSQT_CHANNELS],
        psqt_weights: Option<&[[i32; PSQT_CHANNELS]]>,
    ) -> Self {
        let us = board.turn();
        let them = !us;
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();

        let moved_piece = board.piece_at(from).map(|(p, _)| p).unwrap_or(Piece::Pawn);
        let is_king_move = moved_piece == Piece::King;

        let mut next_white = self.white;
        let mut next_black = self.black;
        let mut next_w_orient = self.white_orient;
        let mut next_b_orient = self.black_orient;

        // --- Handle Moving Side ---
        if is_king_move {
            // Own king moved -> full refresh for moving side
            let (refreshed_acc, new_orient) = Self::refresh_perspective(
                new_board,
                us,
                bias,
                weights,
                psqt_bias,
                psqt_weights,
            );
            if us == Color::White {
                next_white = refreshed_acc;
                next_w_orient = new_orient;
            } else {
                next_black = refreshed_acc;
                next_b_orient = new_orient;
            }
        } else {
            // King did not move for `us` -> incremental update for `us`
            let (target_acc, orient) = if us == Color::White {
                (&mut next_white, &self.white_orient)
            } else {
                (&mut next_black, &self.black_orient)
            };

            let captured = match flag {
                MoveFlag::Capture => board.piece_at(to),
                MoveFlag::EnPassant => Some((Piece::Pawn, them)),
                _ if flag.is_promotion() && flag.is_capture() => board.piece_at(to),
                _ => None,
            };

            let actual_cap_sq = if flag == MoveFlag::EnPassant {
                if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                }
            } else {
                to
            };

            let diff = compute_feature_diff_non_king(
                orient,
                moved_piece,
                us,
                from,
                to,
                captured.map(|(p, c)| (p, c)),
                flag.promotion_piece(),
            );

            // Apply removals
            // Note: For en passant, the captured pawn is on `actual_cap_sq`, which diff handles
            // if we pass actual_cap_sq:
            if flag == MoveFlag::EnPassant {
                // Adjust for en passant capture square
                let c_ep_sq = orient.map_square(actual_cap_sq);
                if let Some(rel_pawn) = super::features::RelativePiece::from_piece(Piece::Pawn, them, orient.perspective) {
                    let ep_feat = orient.feature_index(rel_pawn, c_ep_sq);
                    target_acc.sub_feature(ep_feat, weights, psqt_weights);
                }
                // From pawn removed
                let c_from = orient.map_square(from);
                if let Some(rel_pawn) = super::features::RelativePiece::from_piece(Piece::Pawn, us, orient.perspective) {
                    let from_feat = orient.feature_index(rel_pawn, c_from);
                    target_acc.sub_feature(from_feat, weights, psqt_weights);
                }
                // To pawn added
                let c_to = orient.map_square(to);
                if let Some(rel_pawn) = super::features::RelativePiece::from_piece(Piece::Pawn, us, orient.perspective) {
                    let to_feat = orient.feature_index(rel_pawn, c_to);
                    target_acc.add_feature(to_feat, weights, psqt_weights);
                }
            } else {
                for &rem in &diff.removed {
                    target_acc.sub_feature(rem, weights, psqt_weights);
                }
                for &add in &diff.added {
                    target_acc.add_feature(add, weights, psqt_weights);
                }
            }
        }

        // --- Handle Non-Moving Side (`them`) ---
        // `them` king DID NOT move, so `them` can ALWAYS be updated incrementally!
        {
            let (target_acc, orient) = if them == Color::White {
                (&mut next_white, &self.white_orient)
            } else {
                (&mut next_black, &self.black_orient)
            };

            match flag {
                MoveFlag::KingCastle => {
                    // Enemy king moved
                    let (k_from, k_to) = (from, to);
                    let (r_from, r_to) = if us == Color::White {
                        (Square::H1, Square::F1)
                    } else {
                        (Square::H8, Square::F8)
                    };
                    // King move
                    let rel_k = super::features::RelativePiece::from_piece(Piece::King, us, them).unwrap();
                    target_acc.sub_feature(orient.feature_index(rel_k, orient.map_square(k_from)), weights, psqt_weights);
                    target_acc.add_feature(orient.feature_index(rel_k, orient.map_square(k_to)), weights, psqt_weights);
                    // Rook move
                    let rel_r = super::features::RelativePiece::from_piece(Piece::Rook, us, them).unwrap();
                    target_acc.sub_feature(orient.feature_index(rel_r, orient.map_square(r_from)), weights, psqt_weights);
                    target_acc.add_feature(orient.feature_index(rel_r, orient.map_square(r_to)), weights, psqt_weights);
                }
                MoveFlag::QueenCastle => {
                    let (k_from, k_to) = (from, to);
                    let (r_from, r_to) = if us == Color::White {
                        (Square::A1, Square::D1)
                    } else {
                        (Square::A8, Square::D8)
                    };
                    let rel_k = super::features::RelativePiece::from_piece(Piece::King, us, them).unwrap();
                    target_acc.sub_feature(orient.feature_index(rel_k, orient.map_square(k_from)), weights, psqt_weights);
                    target_acc.add_feature(orient.feature_index(rel_k, orient.map_square(k_to)), weights, psqt_weights);
                    let rel_r = super::features::RelativePiece::from_piece(Piece::Rook, us, them).unwrap();
                    target_acc.sub_feature(orient.feature_index(rel_r, orient.map_square(r_from)), weights, psqt_weights);
                    target_acc.add_feature(orient.feature_index(rel_r, orient.map_square(r_to)), weights, psqt_weights);
                }
                MoveFlag::EnPassant => {
                    let ep_cap_sq = if us == Color::White {
                        unsafe { Square::from_index_unchecked(to.index() - 8) }
                    } else {
                        unsafe { Square::from_index_unchecked(to.index() + 8) }
                    };
                    // Our captured pawn removed
                    let rel_our_pawn = super::features::RelativePiece::from_piece(Piece::Pawn, them, them).unwrap();
                    target_acc.sub_feature(orient.feature_index(rel_our_pawn, orient.map_square(ep_cap_sq)), weights, psqt_weights);
                    // Enemy moving pawn from -> to
                    let rel_enemy_pawn = super::features::RelativePiece::from_piece(Piece::Pawn, us, them).unwrap();
                    target_acc.sub_feature(orient.feature_index(rel_enemy_pawn, orient.map_square(from)), weights, psqt_weights);
                    target_acc.add_feature(orient.feature_index(rel_enemy_pawn, orient.map_square(to)), weights, psqt_weights);
                }
                _ => {
                    let captured = board.piece_at(to);
                    let diff = compute_feature_diff_non_king(
                        orient,
                        moved_piece,
                        us,
                        from,
                        to,
                        captured,
                        flag.promotion_piece(),
                    );
                    for &rem in &diff.removed {
                        target_acc.sub_feature(rem, weights, psqt_weights);
                    }
                    for &add in &diff.added {
                        target_acc.add_feature(add, weights, psqt_weights);
                    }
                }
            }
        }

        Self {
            white: next_white,
            black: next_black,
            white_orient: next_w_orient,
            black_orient: next_b_orient,
        }
    }
}
