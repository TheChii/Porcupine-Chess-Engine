//! Bitboard representation and operations.
//!
//! A bitboard is a 64-bit integer where each bit represents a square on the chess board.
//! This representation enables efficient parallel operations on multiple squares.

mod iter;
mod ops;
mod display;

pub use iter::BitIterator;

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};
use crate::types::Square;

/// A 64-bit bitboard representing squares on the chess board.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// Empty bitboard (no squares set).
    pub const EMPTY: Bitboard = Bitboard(0);

    /// Universe bitboard (all squares set).
    pub const UNIVERSE: Bitboard = Bitboard(!0u64);

    // File masks
    pub const FILE_A: Bitboard = Bitboard(0x0101010101010101);
    pub const FILE_B: Bitboard = Bitboard(0x0202020202020202);
    pub const FILE_C: Bitboard = Bitboard(0x0404040404040404);
    pub const FILE_D: Bitboard = Bitboard(0x0808080808080808);
    pub const FILE_E: Bitboard = Bitboard(0x1010101010101010);
    pub const FILE_F: Bitboard = Bitboard(0x2020202020202020);
    pub const FILE_G: Bitboard = Bitboard(0x4040404040404040);
    pub const FILE_H: Bitboard = Bitboard(0x8080808080808080);

    // Rank masks
    pub const RANK_1: Bitboard = Bitboard(0x00000000000000FF);
    pub const RANK_2: Bitboard = Bitboard(0x000000000000FF00);
    pub const RANK_3: Bitboard = Bitboard(0x0000000000FF0000);
    pub const RANK_4: Bitboard = Bitboard(0x00000000FF000000);
    pub const RANK_5: Bitboard = Bitboard(0x000000FF00000000);
    pub const RANK_6: Bitboard = Bitboard(0x0000FF0000000000);
    pub const RANK_7: Bitboard = Bitboard(0x00FF000000000000);
    pub const RANK_8: Bitboard = Bitboard(0xFF00000000000000);

    // Edge masks
    pub const NOT_FILE_A: Bitboard = Bitboard(!0x0101010101010101);
    pub const NOT_FILE_H: Bitboard = Bitboard(!0x8080808080808080);
    pub const NOT_FILE_AB: Bitboard = Bitboard(!0x0303030303030303);
    pub const NOT_FILE_GH: Bitboard = Bitboard(!0xC0C0C0C0C0C0C0C0);

    // Useful for castling
    pub const BETWEEN_E1_G1: Bitboard = Bitboard(0x60);  // F1 | G1
    pub const BETWEEN_E1_C1: Bitboard = Bitboard(0x0E);  // B1 | C1 | D1
    pub const BETWEEN_E8_G8: Bitboard = Bitboard(0x6000000000000000);
    pub const BETWEEN_E8_C8: Bitboard = Bitboard(0x0E00000000000000);

    /// Create from raw u64.
    #[inline(always)]
    pub const fn new(val: u64) -> Bitboard {
        Bitboard(val)
    }

    /// Create from a single square.
    #[inline(always)]
    pub const fn from_square(sq: Square) -> Bitboard {
        Bitboard(1u64 << sq.index())
    }

    /// Get raw u64 value.
    #[inline(always)]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Check if the bitboard is empty.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if the bitboard is not empty.
    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// Check if there's more than one bit set.
    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 != 0 && (self.0 & (self.0 - 1)) != 0
    }

    /// Check if exactly one bit is set.
    #[inline(always)]
    pub const fn exactly_one(self) -> bool {
        self.0 != 0 && (self.0 & (self.0 - 1)) == 0
    }

    /// Count the number of set bits (population count).
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Get the least significant bit (as a bitboard).
    #[inline(always)]
    pub const fn lsb_bb(self) -> Bitboard {
        Bitboard(self.0 & self.0.wrapping_neg())
    }

    /// Get the least significant bit index as a Square.
    /// Returns None if empty.
    #[inline(always)]
    pub fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(unsafe { Square::from_index_unchecked(self.0.trailing_zeros() as u8) })
        }
    }

    /// Get the least significant bit index, unchecked.
    /// Undefined behavior if empty.
    #[inline(always)]
    pub unsafe fn lsb_unchecked(self) -> Square {
        Square::from_index_unchecked(self.0.trailing_zeros() as u8)
    }

    /// Pop and return the least significant bit.
    /// Returns None if empty.
    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Option<Square> {
        let sq = self.lsb()?;
        self.0 &= self.0 - 1;
        Some(sq)
    }

    /// Pop the least significant bit, unchecked.
    /// Undefined behavior if empty.
    #[inline(always)]
    pub unsafe fn pop_lsb_unchecked(&mut self) -> Square {
        let sq = self.lsb_unchecked();
        self.0 &= self.0 - 1;
        sq
    }

    /// Get the most significant bit index as a Square.
    #[inline(always)]
    pub fn msb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(unsafe { Square::from_index_unchecked(63 - self.0.leading_zeros() as u8) })
        }
    }

    /// Check if a specific square is set.
    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 & (1u64 << sq.index())) != 0
    }

    /// Set a specific square.
    #[inline(always)]
    pub const fn set(self, sq: Square) -> Bitboard {
        Bitboard(self.0 | (1u64 << sq.index()))
    }

    /// Clear a specific square.
    #[inline(always)]
    pub const fn clear(self, sq: Square) -> Bitboard {
        Bitboard(self.0 & !(1u64 << sq.index()))
    }

    /// Toggle a specific square.
    #[inline(always)]
    pub const fn toggle(self, sq: Square) -> Bitboard {
        Bitboard(self.0 ^ (1u64 << sq.index()))
    }

    /// Get an iterator over set squares.
    #[inline]
    pub fn iter(self) -> BitIterator {
        BitIterator::new(self)
    }

    /// Shift north (toward rank 8).
    #[inline(always)]
    pub const fn north(self) -> Bitboard {
        Bitboard(self.0 << 8)
    }

    /// Shift south (toward rank 1).
    #[inline(always)]
    pub const fn south(self) -> Bitboard {
        Bitboard(self.0 >> 8)
    }

    /// Shift east (toward file H).
    #[inline(always)]
    pub const fn east(self) -> Bitboard {
        Bitboard((self.0 << 1) & Self::NOT_FILE_A.0)
    }

    /// Shift west (toward file A).
    #[inline(always)]
    pub const fn west(self) -> Bitboard {
        Bitboard((self.0 >> 1) & Self::NOT_FILE_H.0)
    }

    /// Shift north-east.
    #[inline(always)]
    pub const fn north_east(self) -> Bitboard {
        Bitboard((self.0 << 9) & Self::NOT_FILE_A.0)
    }

    /// Shift north-west.
    #[inline(always)]
    pub const fn north_west(self) -> Bitboard {
        Bitboard((self.0 << 7) & Self::NOT_FILE_H.0)
    }

    /// Shift south-east.
    #[inline(always)]
    pub const fn south_east(self) -> Bitboard {
        Bitboard((self.0 >> 7) & Self::NOT_FILE_A.0)
    }

    /// Shift south-west.
    #[inline(always)]
    pub const fn south_west(self) -> Bitboard {
        Bitboard((self.0 >> 9) & Self::NOT_FILE_H.0)
    }

    /// Flip vertically (swap ranks).
    #[inline(always)]
    pub const fn flip_vertical(self) -> Bitboard {
        Bitboard(self.0.swap_bytes())
    }

    /// Get file mask for a given file index.
    #[inline(always)]
    pub const fn file_mask(file: crate::types::File) -> Bitboard {
        Bitboard(Self::FILE_A.0 << file.index())
    }

    /// Get rank mask for a given rank index.
    #[inline(always)]
    pub const fn rank_mask(rank: crate::types::Rank) -> Bitboard {
        Bitboard(Self::RANK_1.0 << (rank.index() * 8))
    }
}

// Bit operations
impl BitAnd for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl Shl<u8> for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn shl(self, rhs: u8) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}

impl Shr<u8> for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn shr(self, rhs: u8) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard(0x{:016X})", self.0)
    }
}

impl From<u64> for Bitboard {
    #[inline(always)]
    fn from(val: u64) -> Bitboard {
        Bitboard(val)
    }
}

impl From<Bitboard> for u64 {
    #[inline(always)]
    fn from(bb: Bitboard) -> u64 {
        bb.0
    }
}

impl From<Square> for Bitboard {
    #[inline(always)]
    fn from(sq: Square) -> Bitboard {
        Bitboard::from_square(sq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn test_empty_universe() {
        assert!(Bitboard::EMPTY.is_empty());
        assert!(!Bitboard::UNIVERSE.is_empty());
        assert_eq!(Bitboard::EMPTY.count(), 0);
        assert_eq!(Bitboard::UNIVERSE.count(), 64);
    }

    #[test]
    fn test_from_square() {
        let bb = Bitboard::from_square(Square::E4);
        assert_eq!(bb.count(), 1);
        assert!(bb.contains(Square::E4));
        assert!(!bb.contains(Square::E5));
    }

    #[test]
    fn test_lsb() {
        let bb = Bitboard::from_square(Square::E4) | Bitboard::from_square(Square::H8);
        assert_eq!(bb.lsb(), Some(Square::E4));
    }

    #[test]
    fn test_pop_lsb() {
        let mut bb = Bitboard::from_square(Square::A1) | Bitboard::from_square(Square::H8);
        assert_eq!(bb.pop_lsb(), Some(Square::A1));
        assert_eq!(bb.pop_lsb(), Some(Square::H8));
        assert_eq!(bb.pop_lsb(), None);
    }

    #[test]
    fn test_more_than_one() {
        assert!(!Bitboard::EMPTY.more_than_one());
        assert!(!Bitboard::from_square(Square::E4).more_than_one());
        assert!((Bitboard::from_square(Square::E4) | Bitboard::from_square(Square::D4)).more_than_one());
    }

    #[test]
    fn test_shifts() {
        let bb = Bitboard::from_square(Square::E4);
        assert!(bb.north().contains(Square::E5));
        assert!(bb.south().contains(Square::E3));
        assert!(bb.east().contains(Square::F4));
        assert!(bb.west().contains(Square::D4));
    }

    #[test]
    fn test_edge_shifts() {
        // A file piece can't go west
        let a4 = Bitboard::from_square(Square::A4);
        assert!(a4.west().is_empty());

        // H file piece can't go east
        let h4 = Bitboard::from_square(Square::H4);
        assert!(h4.east().is_empty());
    }
}
