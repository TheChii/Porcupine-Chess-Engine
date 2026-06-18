//! Correction history for evaluation adjustment.
//!
//! Tracks the difference between static evaluation and search scores
//! for different pawn structures. Uses this to correct future evaluations.
//!
//! When static eval is consistently higher than search score for a pawn structure,
//! the correction becomes negative (reduces eval). When static eval is consistently
//! lower, the correction becomes positive (increases eval).

use crate::types::Color;

/// Size of the correction history table (power of 2 for fast modulo)
const CORRECTION_SIZE: usize = 16384;

/// Maximum correction value (prevents overcorrection)
const CORRECTION_MAX: i32 = 1024;

#[derive(Clone)]
pub struct CorrectionHistoryTable {
    table: [[i16; CORRECTION_SIZE]; 2],
}

impl CorrectionHistoryTable {
    pub fn new() -> Self {
        Self { table: [[0; CORRECTION_SIZE]; 2] }
    }

    pub fn clear(&mut self) { self.table = [[0; CORRECTION_SIZE]; 2]; }

    #[inline]
    pub fn get(&self, c: Color, h: u64) -> i32 {
        self.table[c.index()][(h as usize) % CORRECTION_SIZE] as i32
    }

    #[inline]
    pub fn update(&mut self, c: Color, h: u64, d: i32, diff: i32) {
        let ci = c.index();
        let i = (h as usize) % CORRECTION_SIZE;
        let b = (diff * d).clamp(-CORRECTION_MAX / 4, CORRECTION_MAX / 4);
        let old = self.table[ci][i] as i32;
        let new = old + b - old * b.abs() / CORRECTION_MAX;
        self.table[ci][i] = new.clamp(-CORRECTION_MAX, CORRECTION_MAX) as i16;
    }

    pub fn age(&mut self) {
        for c in &mut self.table { for e in c { *e /= 2; } }
    }
}

impl Default for CorrectionHistoryTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction_update() {
        let mut table = CorrectionHistoryTable::new();
        let hash = 12345u64;

        // Initial correction should be 0
        assert_eq!(table.get(Color::White, hash), 0);

        // Update with positive diff (search was better than static eval)
        table.update(Color::White, hash, 5, 100);
        assert!(table.get(Color::White, hash) > 0);

        // Different color shouldn't be affected
        assert_eq!(table.get(Color::Black, hash), 0);
    }

    #[test]
    fn test_correction_clamping() {
        let mut table = CorrectionHistoryTable::new();
        let hash = 67890u64;

        // Many large updates should still be clamped
        for _ in 0..100 {
            table.update(Color::White, hash, 10, 500);
        }

        let val = table.get(Color::White, hash);
        assert!(val <= CORRECTION_MAX as i32);
        assert!(val >= -CORRECTION_MAX as i32);
    }
}
