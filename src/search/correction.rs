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

/// Correction history table.
///
/// Indexed by [color][pawn_hash % SIZE] to store correction values for
/// similar pawn structures.
#[derive(Clone)]
pub struct CorrectionHistoryTable {
    table: [[i16; CORRECTION_SIZE]; 2],
}

impl CorrectionHistoryTable {
    /// Create a new empty correction history table.
    pub fn new() -> Self {
        Self {
            table: [[0; CORRECTION_SIZE]; 2],
        }
    }

    /// Clear all correction values.
    pub fn clear(&mut self) {
        self.table = [[0; CORRECTION_SIZE]; 2];
    }

    /// Get the correction value for a pawn hash.
    ///
    /// Returns a value that should be added to the static evaluation.
    /// The returned value is scaled down for direct use.
    #[inline]
    pub fn get(&self, color: Color, pawn_hash: u64) -> i32 {
        let c = color.index();
        let idx = (pawn_hash as usize) % CORRECTION_SIZE;
        i32::from(self.table[c][idx])
    }

    /// Update the correction based on the difference between search score and static eval.
    ///
    /// `diff` = search_score - static_eval
    /// `depth` = search depth (higher depth = more weight)
    #[inline]
    pub fn update(&mut self, color: Color, pawn_hash: u64, depth: i32, diff: i32) {
        let c = color.index();
        let idx = (pawn_hash as usize) % CORRECTION_SIZE;

        // Bonus scaled by depth (higher depth = more reliable)
        let bonus = (diff * depth).clamp(-CORRECTION_MAX / 4, CORRECTION_MAX / 4);

        // Apply gravity update (same as history table)
        let old = i32::from(self.table[c][idx]);
        let new = old + bonus - old * bonus.abs() / CORRECTION_MAX;
        self.table[c][idx] = new.clamp(-CORRECTION_MAX, CORRECTION_MAX) as i16;
    }

    /// Age correction values (divide by 2).
    /// Call at start of new search to give more weight to recent data.
    pub fn age(&mut self) {
        for color in &mut self.table {
            for entry in color {
                *entry /= 2;
            }
        }
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
