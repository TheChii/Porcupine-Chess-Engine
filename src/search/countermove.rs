//! Counter-move heuristic for move ordering.
//!
//! Tracks which move typically refutes the opponent's previous move.
//! Similar to killer moves but indexed by opponent's move rather than ply.

use crate::types::Move;

#[derive(Clone)]
pub struct CounterMoveTable {
    table: [[Option<Move>; 64]; 64],
}

impl CounterMoveTable {
    pub fn new() -> Self { Self { table: [[None; 64]; 64] } }

    #[inline]
    pub fn store(&mut self, m: Move, c: Move) {
        self.table[m.from().index() as usize][m.to().index() as usize] = Some(c);
    }

    #[inline]
    pub fn get(&self, m: Move) -> Option<Move> {
        self.table[m.from().index() as usize][m.to().index() as usize]
    }

    #[inline]
    pub fn is_counter(&self, m: Move, c: Move) -> bool { self.get(m) == Some(c) }

    pub fn clear(&mut self) { self.table = [[None; 64]; 64]; }
}

impl Default for CounterMoveTable {
    fn default() -> Self {
        Self::new()
    }
}
