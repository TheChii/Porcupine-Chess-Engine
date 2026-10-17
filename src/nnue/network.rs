//! NNUE neural network evaluation head: clipped activations and quantized dense layers.
//!
//! # Architecture
//! - Input: 1,024 features from combined White/Black accumulators (side-to-move first).
//! - Feature activation: SCReLU `floor(clamp(u, 0, 255)^2 / 512)` -> [0..126].
//! - Dense 1: 1,024 -> 16 (i8 weights, i32 bias).
//! - Dense 2: 16 -> 32 (i8 weights, i32 bias).
//! - Output: 32 -> 1 (i8 weights, i32 bias).
//! - Optional PSQT skip: 8-channel linear projection combined into final score.

use super::accumulator::{Accumulator, PSQT_CHANNELS, TRANSFORMER_WIDTH};
use crate::types::Score;

pub const DENSE1_IN: usize = TRANSFORMER_WIDTH * 2; // 1,024
pub const DENSE1_OUT: usize = 128;
pub const DENSE2_OUT: usize = 0;

/// Scale factor to convert network raw integer output to engine centipawns.
/// With out_weights scale 128 and activated feature scale 127: 128 * 127 = 16256.
pub const OUTPUT_SCALE: i32 = 16256;

/// Scale factor for PSQT accumulation.
pub const PSQT_SCALE: i32 = 64;

/// SCReLU activation function: `floor(clamp(u, 0, 255)^2 / 512)`.
/// Output range is [0..126].
#[inline(always)]
pub fn screlu(val: i16) -> i32 {
    let clamped = val.clamp(0, 255) as i32;
    (clamped * clamped) / 512
}

/// Clipped ReLU activation function: `clamp(val, 0, 255)`.
#[inline(always)]
pub fn crelu(val: i32) -> i32 {
    val.clamp(0, 255)
}

/// Quantized Dense Network weights and biases.
#[derive(Clone, Debug)]
pub struct NetworkHead {
    /// Dense layer 1 weights: [128][1024]
    pub d1_weights: [[i8; DENSE1_IN]; DENSE1_OUT],
    /// Dense layer 1 biases: [128]
    pub d1_biases: [i32; DENSE1_OUT],

    /// Output layer weights: [128]
    pub out_weights: [i8; DENSE1_OUT],
    /// Output layer bias: 1
    pub out_bias: i32,

    /// Optional PSQT skip linear combination weights: [8]
    pub psqt_weights: [i32; PSQT_CHANNELS],
    /// Whether PSQT skip is active
    pub has_psqt: bool,
}

impl Default for NetworkHead {
    fn default() -> Self {
        Self {
            d1_weights: [[0; DENSE1_IN]; DENSE1_OUT],
            d1_biases: [0; DENSE1_OUT],
            out_weights: [0; DENSE1_OUT],
            out_bias: 0,
            psqt_weights: [0; PSQT_CHANNELS],
            has_psqt: false,
        }
    }
}

impl NetworkHead {
    /// Perform full feedforward inference in safe integer SIMD / scalar arithmetic.
    ///
    /// Evaluates from the perspective of `turn` (side to move).
    /// `stm_acc`: accumulator of the side to move.
    /// `nstm_acc`: accumulator of the other side.
    pub fn evaluate(
        &self,
        stm_acc: &Accumulator,
        nstm_acc: &Accumulator,
    ) -> Score {
        // 1. Feature activations (1,024 elements in [0..126] as u8)
        let mut ft_acts = [0u8; DENSE1_IN];

        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2") {
                unsafe {
                    vec_screlu_1024_avx2(&stm_acc.vals, &nstm_acc.vals, &mut ft_acts);
                }
            } else {
                for i in 0..TRANSFORMER_WIDTH {
                    ft_acts[i] = screlu(stm_acc.vals[i]) as u8;
                    ft_acts[TRANSFORMER_WIDTH + i] = screlu(nstm_acc.vals[i]) as u8;
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..TRANSFORMER_WIDTH {
                ft_acts[i] = screlu(stm_acc.vals[i]) as u8;
                ft_acts[TRANSFORMER_WIDTH + i] = screlu(nstm_acc.vals[i]) as u8;
            }
        }

        // 2. Dense layer 1 (1,024 -> 128)
        let mut d1_acts = [0u8; DENSE1_OUT];

        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2") {
                for i in (0..DENSE1_OUT).step_by(8) {
                    let mut dots = [0i32; 8];
                    unsafe {
                        dense1_8rows_avx2(
                            &self.d1_weights[i..i + 8],
                            &ft_acts,
                            &mut dots,
                        );
                    }
                    for k in 0..8 {
                        let s = self.d1_biases[i + k] + dots[k];
                        d1_acts[i + k] = screlu((s / 32).clamp(0, 255) as i16) as u8;
                    }
                }
            } else {
                for (i, row) in self.d1_weights.iter().enumerate() {
                    let mut sum = self.d1_biases[i];
                    for j in 0..DENSE1_IN {
                        sum += (row[j] as i32) * (ft_acts[j] as i32);
                    }
                    d1_acts[i] = screlu((sum / 32).clamp(0, 255) as i16) as u8;
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for (i, row) in self.d1_weights.iter().enumerate() {
                let mut sum = self.d1_biases[i];
                for j in 0..DENSE1_IN {
                    sum += (row[j] as i32) * (ft_acts[j] as i32);
                }
                d1_acts[i] = screlu((sum / 32).clamp(0, 255) as i16) as u8;
            }
        }

        // 3. Output layer (128 -> 1)
        #[cfg(target_arch = "x86_64")]
        let out_sum = {
            if cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2") {
                let dot = unsafe { output_dot_avx2(&self.out_weights, &d1_acts) };
                self.out_bias + dot
            } else {
                let mut s = self.out_bias;
                for i in 0..DENSE1_OUT {
                    s += (self.out_weights[i] as i32) * (d1_acts[i] as i32);
                }
                s
            }
        };
        #[cfg(not(target_arch = "x86_64"))]
        let out_sum = {
            let mut s = self.out_bias;
            for i in 0..DENSE1_OUT {
                s += (self.out_weights[i] as i32) * (d1_acts[i] as i32);
            }
            s
        };

        // Scale output from dimensionless logits to centipawns (tau = 600 aligns with piece values: N=320, R=500, Q=900)
        let dense_cp = ((out_sum as i64 * 600) / (OUTPUT_SCALE as i64)) as i32;
        let mut final_score = dense_cp;

        // 5. Optional PSQT skip connection
        if self.has_psqt {
            let mut psqt_raw = 0i64;
            for i in 0..PSQT_CHANNELS {
                // Difference between STM and NSTM
                let diff = (stm_acc.psqt[i] - nstm_acc.psqt[i]) as i64;
                psqt_raw += diff * (self.psqt_weights[i] as i64);
            }
            // diff has scale 64, weights has scale 64 -> total divisor is 4096. Convert to centipawns (* 600).
            let psqt_cp = ((psqt_raw * 600) / 4096) as i32;
            final_score += psqt_cp;
        }

        // Clamp to valid centipawn score range
        let cp_score = final_score.clamp(-15000, 15000);
        Score::cp(cp_score)
    }

    /// Fast linear PSQT evaluation from perspective accumulators (~2.5 ns).
    /// Returns None if model does not have PSQT enabled.
    #[inline(always)]
    pub fn evaluate_psqt(
        &self,
        stm_acc: &Accumulator,
        nstm_acc: &Accumulator,
    ) -> Option<Score> {
        if !self.has_psqt {
            return None;
        }
        let mut psqt_raw = 0i64;
        for i in 0..PSQT_CHANNELS {
            let diff = (stm_acc.psqt[i] - nstm_acc.psqt[i]) as i64;
            psqt_raw += diff * (self.psqt_weights[i] as i64);
        }
        let psqt_cp = ((psqt_raw * 600) / 4096) as i32;
        let psqt_score = psqt_cp.clamp(-15000, 15000);
        Some(Score::cp(psqt_score))
    }
}

/// Smooth, monotonic, odd-symmetric rescaling for NNUE scores to eliminate
/// sigmoid saturation compression while preserving strict minimax zero-sum symmetry f(-x) = -f(x).
/// Restores Queen up/down to +/-895 cp, Rook up/down to +/-471 cp.
#[inline(always)]
pub fn rescale_score(score: i32) -> i32 {
    let sign = score.signum();
    let x = score.abs() as i64;
    let res = x + (9 * x * x) / (5 * (x + 60));
    (sign as i64 * res) as i32
}

#[inline(always)]
pub fn rescale_negative_score(score: i32) -> i32 {
    rescale_score(score)
}

// =========================================================================
// AVX2 / SSE Vectorized Kernels
// =========================================================================

/// Vectorized SCReLU for 1,024 elements (512 STM + 512 NSTM).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn vec_screlu_1024_avx2(
    stm_acc: &[i16; TRANSFORMER_WIDTH],
    nstm_acc: &[i16; TRANSFORMER_WIDTH],
    out: &mut [u8; DENSE1_IN],
) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let max_255 = _mm256_set1_epi16(255);

    #[inline(always)]
    unsafe fn process_512(src: *const i16, dst: *mut u8, zero: __m256i, max_255: __m256i) {
        let mut s_ptr = src as *const __m256i;
        let mut d_ptr = dst as *mut u8;

        for _ in 0..16 {
            let v0 = _mm256_loadu_si256(s_ptr);
            let v1 = _mm256_loadu_si256(s_ptr.add(1));

            let c0 = _mm256_min_epi16(_mm256_max_epi16(v0, zero), max_255);
            let c1 = _mm256_min_epi16(_mm256_max_epi16(v1, zero), max_255);

            let sq0 = _mm256_mullo_epi16(c0, c0);
            let sq1 = _mm256_mullo_epi16(c1, c1);

            let res0 = _mm256_srli_epi16(sq0, 9);
            let res1 = _mm256_srli_epi16(sq1, 9);

            let packed = _mm256_packus_epi16(res0, res1);
            let perm = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            _mm256_storeu_si256(d_ptr as *mut __m256i, perm);

            s_ptr = s_ptr.add(2);
            d_ptr = d_ptr.add(32);
        }
    }

    process_512(stm_acc.as_ptr(), out.as_mut_ptr(), zero, max_255);
    process_512(nstm_acc.as_ptr(), out.as_mut_ptr().add(TRANSFORMER_WIDTH), zero, max_255);
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_screlu_1024_avx2(
    stm_acc: &[i16; TRANSFORMER_WIDTH],
    nstm_acc: &[i16; TRANSFORMER_WIDTH],
    out: &mut [u8; DENSE1_IN],
) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let max_255 = _mm256_set1_epi16(255);

    #[inline(always)]
    unsafe fn process_512(src: *const i16, dst: *mut u8, zero: __m256i, max_255: __m256i) {
        let mut s_ptr = src as *const __m256i;
        let mut d_ptr = dst as *mut u8;

        for _ in 0..16 {
            let v0 = _mm256_loadu_si256(s_ptr);
            let v1 = _mm256_loadu_si256(s_ptr.add(1));

            let c0 = _mm256_min_epi16(_mm256_max_epi16(v0, zero), max_255);
            let c1 = _mm256_min_epi16(_mm256_max_epi16(v1, zero), max_255);

            let sq0 = _mm256_mullo_epi16(c0, c0);
            let sq1 = _mm256_mullo_epi16(c1, c1);

            let res0 = _mm256_srli_epi16(sq0, 9);
            let res1 = _mm256_srli_epi16(sq1, 9);

            let packed = _mm256_packus_epi16(res0, res1);
            let perm = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            _mm256_storeu_si256(d_ptr as *mut __m256i, perm);

            s_ptr = s_ptr.add(2);
            d_ptr = d_ptr.add(32);
        }
    }

    process_512(stm_acc.as_ptr(), out.as_mut_ptr(), zero, max_255);
    process_512(nstm_acc.as_ptr(), out.as_mut_ptr().add(TRANSFORMER_WIDTH), zero, max_255);
}

/// AVX2 8-row tiled dot product for Dense 1 (1,024 inputs -> 8 outputs).
/// Loads each 32-byte chunk of activations once and multiplies across 8 weight rows in parallel,
/// reducing activation loads by 87.5% and maximizing AVX2 pipe throughput.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn dense1_8rows_avx2(
    weights: &[[i8; DENSE1_IN]],
    acts: &[u8; DENSE1_IN],
    out_dots: &mut [i32; 8],
) {
    use core::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let mut sum4 = _mm256_setzero_si256();
    let mut sum5 = _mm256_setzero_si256();
    let mut sum6 = _mm256_setzero_si256();
    let mut sum7 = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);

    let mut a_ptr = acts.as_ptr() as *const __m256i;
    let mut w0_ptr = weights[0].as_ptr() as *const __m256i;
    let mut w1_ptr = weights[1].as_ptr() as *const __m256i;
    let mut w2_ptr = weights[2].as_ptr() as *const __m256i;
    let mut w3_ptr = weights[3].as_ptr() as *const __m256i;
    let mut w4_ptr = weights[4].as_ptr() as *const __m256i;
    let mut w5_ptr = weights[5].as_ptr() as *const __m256i;
    let mut w6_ptr = weights[6].as_ptr() as *const __m256i;
    let mut w7_ptr = weights[7].as_ptr() as *const __m256i;

    for _ in 0..32 {
        let a = _mm256_loadu_si256(a_ptr);

        let p0 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w0_ptr));
        let p1 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w1_ptr));
        let p2 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w2_ptr));
        let p3 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w3_ptr));
        let p4 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w4_ptr));
        let p5 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w5_ptr));
        let p6 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w6_ptr));
        let p7 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w7_ptr));

        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));
        sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(p4, ones));
        sum5 = _mm256_add_epi32(sum5, _mm256_madd_epi16(p5, ones));
        sum6 = _mm256_add_epi32(sum6, _mm256_madd_epi16(p6, ones));
        sum7 = _mm256_add_epi32(sum7, _mm256_madd_epi16(p7, ones));

        a_ptr = a_ptr.add(1);
        w0_ptr = w0_ptr.add(1);
        w1_ptr = w1_ptr.add(1);
        w2_ptr = w2_ptr.add(1);
        w3_ptr = w3_ptr.add(1);
        w4_ptr = w4_ptr.add(1);
        w5_ptr = w5_ptr.add(1);
        w6_ptr = w6_ptr.add(1);
        w7_ptr = w7_ptr.add(1);
    }

    let h01 = _mm256_hadd_epi32(sum0, sum1);
    let h23 = _mm256_hadd_epi32(sum2, sum3);
    let h45 = _mm256_hadd_epi32(sum4, sum5);
    let h67 = _mm256_hadd_epi32(sum6, sum7);

    let h0123 = _mm256_hadd_epi32(h01, h23);
    let h4567 = _mm256_hadd_epi32(h45, h67);

    let tot0123 = _mm_add_epi32(
        _mm256_castsi256_si128(h0123),
        _mm256_extracti128_si256(h0123, 1),
    );
    let tot4567 = _mm_add_epi32(
        _mm256_castsi256_si128(h4567),
        _mm256_extracti128_si256(h4567, 1),
    );

    _mm_storeu_si128(out_dots.as_mut_ptr() as *mut __m128i, tot0123);
    _mm_storeu_si128(out_dots.as_mut_ptr().add(4) as *mut __m128i, tot4567);
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
pub unsafe fn dense1_8rows_avx2(
    weights: &[[i8; DENSE1_IN]],
    acts: &[u8; DENSE1_IN],
    out_dots: &mut [i32; 8],
) {
    use core::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let mut sum4 = _mm256_setzero_si256();
    let mut sum5 = _mm256_setzero_si256();
    let mut sum6 = _mm256_setzero_si256();
    let mut sum7 = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);

    let mut a_ptr = acts.as_ptr() as *const __m256i;
    let mut w0_ptr = weights[0].as_ptr() as *const __m256i;
    let mut w1_ptr = weights[1].as_ptr() as *const __m256i;
    let mut w2_ptr = weights[2].as_ptr() as *const __m256i;
    let mut w3_ptr = weights[3].as_ptr() as *const __m256i;
    let mut w4_ptr = weights[4].as_ptr() as *const __m256i;
    let mut w5_ptr = weights[5].as_ptr() as *const __m256i;
    let mut w6_ptr = weights[6].as_ptr() as *const __m256i;
    let mut w7_ptr = weights[7].as_ptr() as *const __m256i;

    for _ in 0..32 {
        let a = _mm256_loadu_si256(a_ptr);

        let p0 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w0_ptr));
        let p1 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w1_ptr));
        let p2 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w2_ptr));
        let p3 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w3_ptr));
        let p4 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w4_ptr));
        let p5 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w5_ptr));
        let p6 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w6_ptr));
        let p7 = _mm256_maddubs_epi16(a, _mm256_loadu_si256(w7_ptr));

        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));
        sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(p4, ones));
        sum5 = _mm256_add_epi32(sum5, _mm256_madd_epi16(p5, ones));
        sum6 = _mm256_add_epi32(sum6, _mm256_madd_epi16(p6, ones));
        sum7 = _mm256_add_epi32(sum7, _mm256_madd_epi16(p7, ones));

        a_ptr = a_ptr.add(1);
        w0_ptr = w0_ptr.add(1);
        w1_ptr = w1_ptr.add(1);
        w2_ptr = w2_ptr.add(1);
        w3_ptr = w3_ptr.add(1);
        w4_ptr = w4_ptr.add(1);
        w5_ptr = w5_ptr.add(1);
        w6_ptr = w6_ptr.add(1);
        w7_ptr = w7_ptr.add(1);
    }

    let h01 = _mm256_hadd_epi32(sum0, sum1);
    let h23 = _mm256_hadd_epi32(sum2, sum3);
    let h45 = _mm256_hadd_epi32(sum4, sum5);
    let h67 = _mm256_hadd_epi32(sum6, sum7);

    let h0123 = _mm256_hadd_epi32(h01, h23);
    let h4567 = _mm256_hadd_epi32(h45, h67);

    let tot0123 = _mm_add_epi32(
        _mm256_castsi256_si128(h0123),
        _mm256_extracti128_si256(h0123, 1),
    );
    let tot4567 = _mm_add_epi32(
        _mm256_castsi256_si128(h4567),
        _mm256_extracti128_si256(h4567, 1),
    );

    _mm_storeu_si128(out_dots.as_mut_ptr() as *mut __m128i, tot0123);
    _mm_storeu_si128(out_dots.as_mut_ptr().add(4) as *mut __m128i, tot4567);
}

/// AVX2 4-row tiled dot product for Dense 1 (1,024 inputs -> 4 outputs).
/// Loads each 32-byte chunk of activations once and multiplies across 4 weight rows in parallel,
/// reducing activation loads by 75% and maximizing AVX2 pipe throughput.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn dense1_4rows_avx2(
    r0: &[i8; DENSE1_IN],
    r1: &[i8; DENSE1_IN],
    r2: &[i8; DENSE1_IN],
    r3: &[i8; DENSE1_IN],
    acts: &[u8; DENSE1_IN],
) -> (i32, i32, i32, i32) {
    use core::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);

    let mut a_ptr = acts.as_ptr() as *const __m256i;
    let mut w0_ptr = r0.as_ptr() as *const __m256i;
    let mut w1_ptr = r1.as_ptr() as *const __m256i;
    let mut w2_ptr = r2.as_ptr() as *const __m256i;
    let mut w3_ptr = r3.as_ptr() as *const __m256i;

    for _ in 0..32 {
        let a = _mm256_loadu_si256(a_ptr);
        let w0 = _mm256_loadu_si256(w0_ptr);
        let w1 = _mm256_loadu_si256(w1_ptr);
        let w2 = _mm256_loadu_si256(w2_ptr);
        let w3 = _mm256_loadu_si256(w3_ptr);

        let p0 = _mm256_maddubs_epi16(a, w0);
        let p1 = _mm256_maddubs_epi16(a, w1);
        let p2 = _mm256_maddubs_epi16(a, w2);
        let p3 = _mm256_maddubs_epi16(a, w3);

        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));

        a_ptr = a_ptr.add(1);
        w0_ptr = w0_ptr.add(1);
        w1_ptr = w1_ptr.add(1);
        w2_ptr = w2_ptr.add(1);
        w3_ptr = w3_ptr.add(1);
    }

    let h01 = _mm256_hadd_epi32(sum0, sum1);
    let h23 = _mm256_hadd_epi32(sum2, sum3);
    let h0123 = _mm256_hadd_epi32(h01, h23);
    let tot = _mm_add_epi32(
        _mm256_castsi256_si128(h0123),
        _mm256_extracti128_si256(h0123, 1),
    );

    let mut res = [0i32; 4];
    _mm_storeu_si128(res.as_mut_ptr() as *mut __m128i, tot);
    (res[0], res[1], res[2], res[3])
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
pub unsafe fn dense1_4rows_avx2(
    r0: &[i8; DENSE1_IN],
    r1: &[i8; DENSE1_IN],
    r2: &[i8; DENSE1_IN],
    r3: &[i8; DENSE1_IN],
    acts: &[u8; DENSE1_IN],
) -> (i32, i32, i32, i32) {
    use core::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);

    let mut a_ptr = acts.as_ptr() as *const __m256i;
    let mut w0_ptr = r0.as_ptr() as *const __m256i;
    let mut w1_ptr = r1.as_ptr() as *const __m256i;
    let mut w2_ptr = r2.as_ptr() as *const __m256i;
    let mut w3_ptr = r3.as_ptr() as *const __m256i;

    for _ in 0..32 {
        let a = _mm256_loadu_si256(a_ptr);
        let w0 = _mm256_loadu_si256(w0_ptr);
        let w1 = _mm256_loadu_si256(w1_ptr);
        let w2 = _mm256_loadu_si256(w2_ptr);
        let w3 = _mm256_loadu_si256(w3_ptr);

        let p0 = _mm256_maddubs_epi16(a, w0);
        let p1 = _mm256_maddubs_epi16(a, w1);
        let p2 = _mm256_maddubs_epi16(a, w2);
        let p3 = _mm256_maddubs_epi16(a, w3);

        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));

        a_ptr = a_ptr.add(1);
        w0_ptr = w0_ptr.add(1);
        w1_ptr = w1_ptr.add(1);
        w2_ptr = w2_ptr.add(1);
        w3_ptr = w3_ptr.add(1);
    }

    let h01 = _mm256_hadd_epi32(sum0, sum1);
    let h23 = _mm256_hadd_epi32(sum2, sum3);
    let h0123 = _mm256_hadd_epi32(h01, h23);
    let tot = _mm_add_epi32(
        _mm256_castsi256_si128(h0123),
        _mm256_extracti128_si256(h0123, 1),
    );

    let mut res = [0i32; 4];
    _mm_storeu_si128(res.as_mut_ptr() as *mut __m128i, tot);
    (res[0], res[1], res[2], res[3])
}

/// AVX2 1,024-element dot product for Dense 1 (i8 weights * u8 activations) unrolled 4x with 4 accumulators.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn dense1_dot_avx2(row: &[i8; DENSE1_IN], acts: &[u8; DENSE1_IN]) -> i32 {
    use core::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);

    let mut w_ptr = row.as_ptr() as *const __m256i;
    let mut a_ptr = acts.as_ptr() as *const __m256i;

    for _ in 0..8 {
        let w0 = _mm256_loadu_si256(w_ptr);
        let a0 = _mm256_loadu_si256(a_ptr);
        let p0 = _mm256_maddubs_epi16(a0, w0);
        let s0 = _mm256_madd_epi16(p0, ones);
        sum0 = _mm256_add_epi32(sum0, s0);

        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let a1 = _mm256_loadu_si256(a_ptr.add(1));
        let p1 = _mm256_maddubs_epi16(a1, w1);
        let s1 = _mm256_madd_epi16(p1, ones);
        sum1 = _mm256_add_epi32(sum1, s1);

        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let a2 = _mm256_loadu_si256(a_ptr.add(2));
        let p2 = _mm256_maddubs_epi16(a2, w2);
        let s2 = _mm256_madd_epi16(p2, ones);
        sum2 = _mm256_add_epi32(sum2, s2);

        let w3 = _mm256_loadu_si256(w_ptr.add(3));
        let a3 = _mm256_loadu_si256(a_ptr.add(3));
        let p3 = _mm256_maddubs_epi16(a3, w3);
        let s3 = _mm256_madd_epi16(p3, ones);
        sum3 = _mm256_add_epi32(sum3, s3);

        w_ptr = w_ptr.add(4);
        a_ptr = a_ptr.add(4);
    }

    let sum_01 = _mm256_add_epi32(sum0, sum1);
    let sum_23 = _mm256_add_epi32(sum2, sum3);
    let total_sum = _mm256_add_epi32(sum_01, sum_23);
    hsum_epi32_avx2(total_sum)
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
pub unsafe fn dense1_dot_avx2(row: &[i8; DENSE1_IN], acts: &[u8; DENSE1_IN]) -> i32 {
    use core::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);

    let mut w_ptr = row.as_ptr() as *const __m256i;
    let mut a_ptr = acts.as_ptr() as *const __m256i;

    for _ in 0..8 {
        let w0 = _mm256_loadu_si256(w_ptr);
        let a0 = _mm256_loadu_si256(a_ptr);
        let p0 = _mm256_maddubs_epi16(a0, w0);
        let s0 = _mm256_madd_epi16(p0, ones);
        sum0 = _mm256_add_epi32(sum0, s0);

        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let a1 = _mm256_loadu_si256(a_ptr.add(1));
        let p1 = _mm256_maddubs_epi16(a1, w1);
        let s1 = _mm256_madd_epi16(p1, ones);
        sum1 = _mm256_add_epi32(sum1, s1);

        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let a2 = _mm256_loadu_si256(a_ptr.add(2));
        let p2 = _mm256_maddubs_epi16(a2, w2);
        let s2 = _mm256_madd_epi16(p2, ones);
        sum2 = _mm256_add_epi32(sum2, s2);

        let w3 = _mm256_loadu_si256(w_ptr.add(3));
        let a3 = _mm256_loadu_si256(a_ptr.add(3));
        let p3 = _mm256_maddubs_epi16(a3, w3);
        let s3 = _mm256_madd_epi16(p3, ones);
        sum3 = _mm256_add_epi32(sum3, s3);

        w_ptr = w_ptr.add(4);
        a_ptr = a_ptr.add(4);
    }

    let sum_01 = _mm256_add_epi32(sum0, sum1);
    let sum_23 = _mm256_add_epi32(sum2, sum3);
    let total_sum = _mm256_add_epi32(sum_01, sum_23);
    hsum_epi32_avx2(total_sum)
}

/// AVX2 128-element dot product for Output layer (i8 weights * u8 activations).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn output_dot_avx2(weights: &[i8; DENSE1_OUT], acts: &[u8; DENSE1_OUT]) -> i32 {
    use core::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let mut sum = _mm256_setzero_si256();
    let mut w_ptr = weights.as_ptr() as *const __m256i;
    let mut a_ptr = acts.as_ptr() as *const __m256i;

    for _ in 0..4 {
        let w = _mm256_loadu_si256(w_ptr);
        let a = _mm256_loadu_si256(a_ptr);
        let p = _mm256_maddubs_epi16(a, w);
        let s = _mm256_madd_epi16(p, ones);
        sum = _mm256_add_epi32(sum, s);
        w_ptr = w_ptr.add(1);
        a_ptr = a_ptr.add(1);
    }
    hsum_epi32_avx2(sum)
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
pub unsafe fn output_dot_avx2(weights: &[i8; DENSE1_OUT], acts: &[u8; DENSE1_OUT]) -> i32 {
    use core::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let mut sum = _mm256_setzero_si256();
    let mut w_ptr = weights.as_ptr() as *const __m256i;
    let mut a_ptr = acts.as_ptr() as *const __m256i;

    for _ in 0..4 {
        let w = _mm256_loadu_si256(w_ptr);
        let a = _mm256_loadu_si256(a_ptr);
        let p = _mm256_maddubs_epi16(a, w);
        let s = _mm256_madd_epi16(p, ones);
        sum = _mm256_add_epi32(sum, s);
        w_ptr = w_ptr.add(1);
        a_ptr = a_ptr.add(1);
    }
    hsum_epi32_avx2(sum)
}

/// Horizontal sum of 8 32-bit integers in __m256i.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn hsum_epi32_avx2(v: core::arch::x86_64::__m256i) -> i32 {
    use core::arch::x86_64::*;
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b0000_0001);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_avx2(v: core::arch::x86_64::__m256i) -> i32 {
    use core::arch::x86_64::*;
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b0000_0001);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screlu_bounds() {
        assert_eq!(screlu(-100), 0);
        assert_eq!(screlu(0), 0);
        assert_eq!(screlu(16), (16 * 16) / 512); // 0
        assert_eq!(screlu(32), (32 * 32) / 512); // 2
        assert_eq!(screlu(128), (128 * 128) / 512); // 32
        assert_eq!(screlu(255), (255 * 255) / 512); // 126
        assert_eq!(screlu(500), (255 * 255) / 512); // 126
    }

    #[test]
    fn test_zero_eval() {
        let net = NetworkHead::default();
        let acc1 = Accumulator::default();
        let acc2 = Accumulator::default();
        let score = net.evaluate(&acc1, &acc2);
        assert_eq!(score.raw(), 0);
    }
}
