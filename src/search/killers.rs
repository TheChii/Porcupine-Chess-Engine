//! Killer moves heuristic for move ordering.
//!
//! Killer moves are quiet moves that caused beta cutoffs at the same ply.
//! They are likely to be good moves and should be searched early.

use crate::types::{Move, Ply, MAX_PLY};

/// Number of killer move slots per ply
const NUM_KILLERS: usize = 2;

#[derive(Clone)]
pub struct KillerTable {
    killers: [[Option<Move>; NUM_KILLERS]; MAX_PLY as usize],
}

impl KillerTable {
    pub fn new() -> Self { Self { killers: [[None; NUM_KILLERS]; MAX_PLY as usize] } }

    #[inline]
    pub fn store(&mut self, p: Ply, m: Move) {
        let i = p.raw() as usize;
        if i >= MAX_PLY as usize || self.killers[i][0] == Some(m) { return; }
        self.killers[i][1] = self.killers[i][0];
        self.killers[i][0] = Some(m);
    }

    #[inline]
    pub fn get(&self, p: Ply) -> [Option<Move>; NUM_KILLERS] {
        let i = p.raw() as usize;
        if i >= MAX_PLY as usize { [None; NUM_KILLERS] } else { self.killers[i] }
    }

    #[inline]
    pub fn is_killer(&self, p: Ply, m: Move) -> Option<usize> {
        let ks = self.get(p);
        if ks[0] == Some(m) { Some(0) } else if ks[1] == Some(m) { Some(1) } else { None }
    }

    pub fn clear(&mut self) {
        for pk in &mut self.killers { *pk = [None; NUM_KILLERS]; }
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
    use movegen::{MoveFlag, Square};

    #[test]
    fn test_killer_store_and_get() {
        let mut table = KillerTable::new();
        let ply = Ply::new(5);

        // Create test moves
        let mv1 = Move::new(Square::E2, Square::E4, MoveFlag::DoublePawnPush);
        let mv2 = Move::new(Square::D2, Square::D4, MoveFlag::DoublePawnPush);
        let mv3 = Move::new(Square::G1, Square::F3, MoveFlag::Quiet);

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
