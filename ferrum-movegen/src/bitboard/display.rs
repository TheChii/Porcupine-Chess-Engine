//! Display utilities for debugging bitboards.

use super::Bitboard;
use core::fmt;

impl Bitboard {
    /// Pretty-print the bitboard as an 8x8 grid.
    pub fn pretty_print(&self) -> String {
        let mut result = String::with_capacity(200);
        result.push_str("  +---+---+---+---+---+---+---+---+\n");
        
        for rank in (0..8).rev() {
            result.push_str(&format!("{} |", rank + 1));
            for file in 0..8 {
                let sq = rank * 8 + file;
                let bit = (self.0 >> sq) & 1;
                if bit == 1 {
                    result.push_str(" X |");
                } else {
                    result.push_str("   |");
                }
            }
            result.push('\n');
            result.push_str("  +---+---+---+---+---+---+---+---+\n");
        }
        result.push_str("    a   b   c   d   e   f   g   h\n");
        result
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pretty_print())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn test_pretty_print() {
        let bb = Bitboard::from_square(Square::E4);
        let output = bb.pretty_print();
        assert!(output.contains("X"));
    }
}
