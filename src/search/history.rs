//! History heuristic for move ordering.
//!
//! Tracks which quiet moves cause beta cutoffs and uses
//! accumulated scores to order moves better in future searches.

use crate::types::{Color, Move};

#[derive(Clone)]
pub struct HistoryTable {
    table: [[[i32; 64]; 64]; 2],
}

impl HistoryTable {
    pub fn new() -> Self { Self { table: [[[0; 64]; 64]; 2] } }

    #[inline]
    pub fn get(&self, clr: Color, mv: Move) -> i32 {
        self.table[clr.index()][mv.from().index() as usize][mv.to().index() as usize]
    }

    #[inline]
    pub fn update(&mut self, clr: Color, mv: Move, sc: i32) {
        let ci = clr.index();
        let f = mv.from().index() as usize;
        let t = mv.to().index() as usize;
        let old = self.table[ci][f][t];
        let max = 16384; 
        let cb = sc.clamp(-2000, 2000); 
        self.table[ci][f][t] = old + cb - old * cb.abs() / max;
    }

    pub fn update_on_cutoff(&mut self, c: Color, bm: Move, d: i32, oqs: &[Move]) {
        let b = (16 * d * d).min(2000);
        self.update(c, bm, b);
        for &m in oqs { if m != bm { self.update(c, m, -b); } }
    }

    pub fn clear(&mut self) { self.table = [[[0; 64]; 64]; 2]; }

    pub fn age(&mut self) {
        for c in &mut self.table { for f in c { for t in f { *t /= 2; } } }
    }
}

impl Default for HistoryTable {
    fn default() -> Self { Self::new() }
}
