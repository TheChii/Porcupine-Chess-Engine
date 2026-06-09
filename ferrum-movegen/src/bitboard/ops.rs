//! Additional bitboard operations.

use super::Bitboard;

impl Bitboard {
    /// Fill north (kogge-stone style).
    #[inline]
    pub const fn fill_north(self) -> Bitboard {
        let mut bb = self.0;
        bb |= bb << 8;
        bb |= bb << 16;
        bb |= bb << 32;
        Bitboard(bb)
    }

    /// Fill south (kogge-stone style).
    #[inline]
    pub const fn fill_south(self) -> Bitboard {
        let mut bb = self.0;
        bb |= bb >> 8;
        bb |= bb >> 16;
        bb |= bb >> 32;
        Bitboard(bb)
    }

    /// Fill file (both directions).
    #[inline]
    pub const fn fill_file(self) -> Bitboard {
        let mut bb = self.0;
        bb |= bb << 8;
        bb |= bb >> 8;
        bb |= bb << 16;
        bb |= bb >> 16;
        bb |= bb << 32;
        bb |= bb >> 32;
        Bitboard(bb)
    }

    /// Get adjacent files (neighbors).
    #[inline]
    pub const fn adjacent_files(self) -> Bitboard {
        let fill = self.fill_file();
        Bitboard(((fill.0 << 1) & !Self::FILE_A.0) | ((fill.0 >> 1) & !Self::FILE_H.0))
    }

    /// Spread by one square in all directions (king-style).
    #[inline]
    pub const fn spread(self) -> Bitboard {
        let bb = self.0;
        let lr = ((bb << 1) & !Self::FILE_A.0) | ((bb >> 1) & !Self::FILE_H.0);
        let all = bb | lr;
        Bitboard(all | (all << 8) | (all >> 8))
    }

    /// Double pawn push targets for white.
    #[inline]
    pub const fn white_double_push_targets(self, empty: Bitboard) -> Bitboard {
        let single = Bitboard((self.0 << 8) & empty.0);
        Bitboard((single.0 << 8) & empty.0 & Self::RANK_4.0)
    }

    /// Double pawn push targets for black.
    #[inline]
    pub const fn black_double_push_targets(self, empty: Bitboard) -> Bitboard {
        let single = Bitboard((self.0 >> 8) & empty.0);
        Bitboard((single.0 >> 8) & empty.0 & Self::RANK_5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn test_fill_north() {
        let bb = Bitboard::from_square(Square::E4);
        let filled = bb.fill_north();
        assert!(filled.contains(Square::E4));
        assert!(filled.contains(Square::E5));
        assert!(filled.contains(Square::E8));
        assert!(!filled.contains(Square::E3));
    }

    #[test]
    fn test_fill_south() {
        let bb = Bitboard::from_square(Square::E4);
        let filled = bb.fill_south();
        assert!(filled.contains(Square::E4));
        assert!(filled.contains(Square::E1));
        assert!(!filled.contains(Square::E5));
    }

    #[test]
    fn test_spread() {
        let bb = Bitboard::from_square(Square::E4);
        let spread = bb.spread();
        assert_eq!(spread.count(), 9); // Center + 8 surrounding
        assert!(spread.contains(Square::E4));
        assert!(spread.contains(Square::D3));
        assert!(spread.contains(Square::F5));
    }
}
