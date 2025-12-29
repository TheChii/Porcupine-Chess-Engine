//! History heuristic for move ordering.
//!
//! Tracks which quiet moves cause beta cutoffs and uses
//! accumulated scores to order moves better in future searches.

use crate::types::{Move, Color};

/// History table: [color][from_sq][to_sq] -> score
#[derive(Clone)]
pub struct HistoryTable {
    table: [[[i32; 64]; 64]; 2],
}

impl HistoryTable {
    /// Create a new empty history table
    pub fn new() -> Self {
        Self {
            table: [[[0; 64]; 64]; 2],
        }
    }

    /// Get history score for a move
    #[inline]
    pub fn get(&self, color: Color, mv: Move) -> i32 {
        let c = color.index();
        let from = mv.from().index() as usize;
        let to = mv.to().index() as usize;
        self.table[c][from][to]
    }

    /// Update history score for a move that caused cutoff
    /// Bonus is typically depth * depth
    #[inline]
    pub fn update(&mut self, color: Color, mv: Move, bonus: i32) {
        let c = color.index();
        let from = mv.from().index() as usize;
        let to = mv.to().index() as usize;
        
        // Gravity formula: prevents scores from growing unbounded
        // new_score = old_score + bonus - (old_score * |bonus| / max)
        let old = self.table[c][from][to];
        let max = 16384;
        let clamped_bonus = bonus.clamp(-max, max);
        self.table[c][from][to] = old + clamped_bonus - old * clamped_bonus.abs() / max;
    }

    /// Apply bonus to move that caused cutoff, penalty to other quiet moves
    pub fn update_on_cutoff(&mut self, color: Color, best_move: Move, depth: i32, other_quiets: &[Move]) {
        let bonus = depth * depth;
        
        // Bonus for the move that caused cutoff
        self.update(color, best_move, bonus);
        
        // Penalty for quiet moves that didn't cause cutoff
        for &m in other_quiets {
            if m != best_move {
                self.update(color, m, -bonus);
            }
        }
    }

    /// Clear all history (call at start of new game, not new search)
    pub fn clear(&mut self) {
        self.table = [[[0; 64]; 64]; 2];
    }

    /// Age history scores (call at start of new search)
    /// Divides all scores by 2 to give more weight to recent moves
    pub fn age(&mut self) {
        for color in &mut self.table {
            for from in color {
                for to in from {
                    *to /= 2;
                }
            }
        }
    }
}

impl Default for HistoryTable {
    fn default() -> Self {
        Self::new()
    }
}
