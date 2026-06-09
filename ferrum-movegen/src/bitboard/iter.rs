//! Iterator over set bits in a Bitboard.

use super::Bitboard;
use crate::types::Square;

/// Iterator that yields squares for each set bit in a Bitboard.
#[derive(Clone, Copy)]
pub struct BitIterator {
    bits: Bitboard,
}

impl BitIterator {
    /// Create a new iterator from a bitboard.
    #[inline(always)]
    pub const fn new(bits: Bitboard) -> BitIterator {
        BitIterator { bits }
    }
}

impl Iterator for BitIterator {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        self.bits.pop_lsb()
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.bits.count() as usize;
        (count, Some(count))
    }

    #[inline(always)]
    fn count(self) -> usize {
        self.bits.count() as usize
    }
}

impl ExactSizeIterator for BitIterator {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitIterator;

    #[inline(always)]
    fn into_iter(self) -> BitIterator {
        BitIterator::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iterator() {
        let bb = Bitboard::from_square(Square::A1) 
               | Bitboard::from_square(Square::E4) 
               | Bitboard::from_square(Square::H8);

        let squares: Vec<_> = bb.iter().collect();
        assert_eq!(squares.len(), 3);
        assert!(squares.contains(&Square::A1));
        assert!(squares.contains(&Square::E4));
        assert!(squares.contains(&Square::H8));
    }

    #[test]
    fn test_exact_size() {
        let bb = Bitboard::from_square(Square::A1) | Bitboard::from_square(Square::H8);
        let iter = bb.iter();
        assert_eq!(iter.len(), 2);
    }
}
