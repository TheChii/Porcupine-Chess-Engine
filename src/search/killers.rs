//! Killer moves heuristic for move ordering.
//!
//! Killer moves are quiet moves that caused beta cutoffs at the same ply.
//! They are likely to be good moves and should be searched early.

use crate::types::{Move, Ply, MAX_PLY};

/// Number of killer move slots per ply
const NUM_KILLERS: usize = 2;

/// Killer moves table
/// Stores 2 killer moves per ply that caused beta cutoffs
#[derive(Clone)]
pub struct KillerTable {
    killers: [[Option<Move>; NUM_KILLERS]; MAX_PLY as usize],
}

impl KillerTable {
    /// Create a new empty killer table
    pub fn new() -> Self {
        Self {
            killers: [[None; NUM_KILLERS]; MAX_PLY as usize],
        }
    }

    /// Store a killer move at the given ply
    /// Shifts existing killers (slot 1 becomes slot 0) if different
    #[inline]
    pub fn store(&mut self, ply: Ply, mv: Move) {
        let idx = ply.raw() as usize;
        if idx >= MAX_PLY as usize {
            return;
        }

        // Don't store if it's already killer 0
        if self.killers[idx][0] == Some(mv) {
            return;
        }

        // Shift killer 0 to killer 1, store new move as killer 0
        self.killers[idx][1] = self.killers[idx][0];
        self.killers[idx][0] = Some(mv);
    }

    /// Get killer moves at the given ply
    #[inline]
    pub fn get(&self, ply: Ply) -> [Option<Move>; NUM_KILLERS] {
        let idx = ply.raw() as usize;
        if idx >= MAX_PLY as usize {
            return [None; NUM_KILLERS];
        }
        self.killers[idx]
    }

    /// Check if a move is a killer at the given ply
    #[inline]
    pub fn is_killer(&self, ply: Ply, mv: Move) -> Option<usize> {
        let killers = self.get(ply);
        if killers[0] == Some(mv) {
            Some(0)
        } else if killers[1] == Some(mv) {
            Some(1)
        } else {
            None
        }
    }

    /// Clear all killers (call at start of new search)
    pub fn clear(&mut self) {
        for ply_killers in &mut self.killers {
            *ply_killers = [None; NUM_KILLERS];
        }
    }
}

impl Default for KillerTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_killer_store_and_get() {
        let mut table = KillerTable::new();
        let ply = Ply::new(5);
        
        // Create test moves
        let mv1 = Move::new(chess::Square::E2, chess::Square::E4, None);
        let mv2 = Move::new(chess::Square::D2, chess::Square::D4, None);
        let mv3 = Move::new(chess::Square::G1, chess::Square::F3, None);

        // Store first killer
        table.store(ply, mv1);
        assert_eq!(table.get(ply)[0], Some(mv1));
        assert_eq!(table.get(ply)[1], None);

        // Store second killer (different move)
        table.store(ply, mv2);
        assert_eq!(table.get(ply)[0], Some(mv2));
        assert_eq!(table.get(ply)[1], Some(mv1));

        // Store third killer (shifts again)
        table.store(ply, mv3);
        assert_eq!(table.get(ply)[0], Some(mv3));
        assert_eq!(table.get(ply)[1], Some(mv2));

        // Storing same killer again shouldn't change anything
        table.store(ply, mv3);
        assert_eq!(table.get(ply)[0], Some(mv3));
        assert_eq!(table.get(ply)[1], Some(mv2));
    }
}
